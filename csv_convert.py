#!/usr/bin/env python3
import os
import re
import sys
import io
import csv
import json
import argparse
from datetime import date, timedelta
from email import policy
from email.parser import BytesParser

from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import pdfplumber
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

# --- Compute sensible defaults ---
TODAY = date.today()
DEFAULT_INVOICE_DATE = TODAY.strftime("%m/%d/%Y")
DEFAULT_DUE_DATE     = (TODAY + timedelta(days=30)).strftime("%m/%d/%Y")
DEFAULT_ORG          = TODAY.strftime("%m/%d/%Y")
DEFAULT_OBJECT       = (TODAY + timedelta(days=30)).strftime("%m/%d/%Y")

# --- Load public Donut base model & processor ---
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", use_fast=False)
model     = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# --- Pre-build our JSON‐schema prompt ---
# Donut expects a “task prompt” in XML-like tags
SCHEMA_PROMPT = (
    "<s_invoice>"
      "<s_vendor_number></s_vendor_number>"
      "<s_invoice_number></s_invoice_number>"
      "<s_invoice_total></s_invoice_total>"
      "<s_invoice_net_amount></s_invoice_net_amount>"
    "</s_invoice>"
)
PROMPT_IDS = processor.tokenizer(
    SCHEMA_PROMPT, add_special_tokens=False, return_tensors="pt"
).input_ids

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text = ""
    reader = PdfReader(io.BytesIO(b))
    for page in reader.pages:
        if (p := page.extract_text()):
            text += p
    if len(text.strip()) < 20:
        try:
            for img in convert_from_bytes(b):
                text += pytesseract.image_to_string(img)
        except Exception:
            pass
    return text

def parse_invoice_with_donut(pdf_bytes: bytes) -> dict:
    """Render the first page as image, pass through Donut with our schema prompt,
       and return the parsed JSON."""
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
    if not images:
        return {}
    img = images[0].convert("RGB")
    enc = processor(img, return_tensors="pt")
    # supply our pre-built prompt IDs as the initial decoder input
    outputs = model.generate(
        **enc,
        decoder_input_ids=PROMPT_IDS,
        max_length=512
    )
    raw = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

def extract_vendor_number(text: str, source_file: str, donut_json: dict) -> str:
    # 1) JSON schema field
    if v := donut_json.get("vendor_number"):
        return v
    # 2) explicit label
    if m := re.search(r"(?:vendor|acct(?:ount)?)\s*(?:#|number)[:\s]*([0-9]{6,8})",
                      text, re.IGNORECASE):
        return m.group(1).zfill(8)
    # 3) standalone 8-digit
    if fb := re.findall(r"\b\d{8}\b", text):
        return fb[0]
    # 4) filename fallback
    return os.path.splitext(os.path.basename(source_file))[0]

def extract_invoice_number(text: str, source_file: str, donut_json: dict) -> str:
    if v := donut_json.get("invoice_number") or donut_json.get("invoice_id") or donut_json.get("invoice_no"):
        return v
    if m := re.search(r"(?:invoice\s*(?:no\.?|#|number)\s*[:\-]?\s*)([A-Z0-9\-/]{4,20})",
                      text, re.IGNORECASE):
        return m.group(1).strip()
    if m2 := re.search(r"\d{1,2}/\d{1,2}/\d{2,4}\s+([A-Z0-9\-]{5,})", text):
        return m2.group(1)
    if fb := re.findall(r"\b[0-9]{2}[A-Z]{3}[0-9]{6,}\b", text):
        return fb[0]
    return os.path.splitext(os.path.basename(source_file))[0]

def extract_financial_details(text: str, pdf_bytes: bytes, donut_json: dict) -> dict:
    details = {}
    # 1) from JSON schema
    if jt := donut_json.get("invoice_total"):
        details["invoice_total"] = jt
    if jn := donut_json.get("invoice_net_amount"):
        details["invoice_net_amount"] = jn

    # 2) regex fallback for total
    amt_re = r"(?<!\d)(?:USD|US\$|\$|£|€)\s*([\d,]+(?:\.\d{2}))"
    if "invoice_total" not in details:
        nums = [float(a.replace(",","")) for a in re.findall(amt_re, text)]
        if nums:
            details["invoice_total"] = f"{max(nums):.2f}"
    if "invoice_net_amount" not in details:
        nums = [float(a.replace(",","")) for a in re.findall(amt_re, text)]
        if len(nums) > 1:
            details["invoice_net_amount"] = f"{sorted(nums, reverse=True)[1]:.2f}"
        elif nums:
            details["invoice_net_amount"] = f"{nums[0]:.2f}"

    # 3) table fallback for invoice_total
    if "invoice_total" not in details:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    tables = p.extract_tables()
                    if tables:
                        last = tables[0][-1][-1]
                        if last and re.match(r"[\d,]+\.\d{2}", last):
                            details["invoice_total"] = last.replace(",", "")
                            break
        except:
            pass

    return details

def extract_records_from_eml(eml_path: str) -> list[dict]:
    records = []
    with open(eml_path, "rb") as fp:
        msg = BytesParser(policy=policy.default).parse(fp)

    for part in msg.walk():
        # guard against None filename
        fn = part.get_filename() or ""
        if part.get_content_type() != "application/pdf" and not fn.lower().endswith(".pdf"):
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue

        text       = extract_text_from_pdf_bytes(payload)
        donut_json = parse_invoice_with_donut(payload)

        records.append({
            "vendor_number":      extract_vendor_number(text, eml_path, donut_json),
            "invoice_number":     extract_invoice_number(text, eml_path, donut_json),
            "invoice_total":      extract_financial_details(text, payload, donut_json).get("invoice_total",""),
            "invoice_net_amount": extract_financial_details(text, payload, donut_json).get("invoice_net_amount",""),
        })

    return records

def write_munis_csv(records: list[dict], out_path: str, args):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Row Type","Vendor Number","Remit Number","Invoice Number",
            "Invoice Date","Invoice Due Date","Invoice Total","Invoice Net Amount",
            "PO Fiscal Year","PO Number","Include Documentation","Separate Check",
            "Contract Number","Invoice Description"
        ])
        for rec in records:
            w.writerow([
                "1", rec["vendor_number"], args.remit_number, rec["invoice_number"],
                args.invoice_date, args.invoice_due_date,
                rec["invoice_total"], rec["invoice_net_amount"],
                args.po_fiscal_year, args.po_number,
                args.include_documentation, args.separate_check,
                args.contract_number, args.invoice_description
            ])
            w.writerow([
                "2", rec["vendor_number"], args.sequence_start, rec["invoice_number"],
                args.default_org, args.default_object, args.project,
                rec["invoice_total"], args.po_line_number,
                "","","","", args.detail_description
            ])

def main():
    p = argparse.ArgumentParser(description="Produce Munis CSV from PDFs in EMLs")
    p.add_argument("-f","--folder",      default=".", help="Directory with .eml files")
    p.add_argument("-o","--output",      default="munis_import.csv", help="CSV filename")
    p.add_argument("--default-org",      default=DEFAULT_ORG)
    p.add_argument("--default-object",   default=DEFAULT_OBJECT)
    p.add_argument("--remit-number",     default="0")
    p.add_argument("--invoice-date",     default=DEFAULT_INVOICE_DATE)
    p.add_argument("--invoice-due-date", default=DEFAULT_DUE_DATE)
    p.add_argument("--po-fiscal-year",   default="")
    p.add_argument("--po-number",        default="")
    p.add_argument("--include-documentation", default="")
    p.add_argument("--separate-check",   default="")
    p.add_argument("--contract-number",  default="")
    p.add_argument("--invoice-description", default="")
    p.add_argument("--project",          default="")
    p.add_argument("--po-line-number",   default="")
    p.add_argument("--sequence-start",   default="1")
    p.add_argument("--detail-description", default="")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Error: folder not found: {args.folder}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eml_files = []
    for root, _, files in os.walk(args.folder):
        for fn in files:
            if fn.lower().endswith(".eml"):
                eml_files.append(os.path.join(root, fn))
    eml_files.sort()

    if eml_files:
        print("Found .eml files:")
        for f in eml_files:
            print("  ", f)
    else:
        print("No .eml files in folder.")

    records = []
    for eml in eml_files:
        recs = extract_records_from_eml(eml)
        print(f"→ {len(recs)} PDF attachment(s) in {os.path.basename(eml)}")
        records.extend(recs)

    if not records:
        sys.exit("No PDF attachments found.")

    seen, unique = set(), []
    for rec in records:
        key = (rec["vendor_number"], rec["invoice_number"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    out_path = os.path.join(script_dir, args.output)
    write_munis_csv(unique, out_path, args)
    print(f"✅ Wrote {len(unique)} invoices to {out_path}")

if __name__ == "__main__":
    main()