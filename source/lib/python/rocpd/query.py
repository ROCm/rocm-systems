#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

import os
import sys

from typing import Union, Tuple, List, Optional
from datetime import datetime

from . import output_config
from . import libpyrocpd
from .importer import RocpdImportData
from .time_window import apply_time_window


def export_sqlite_query(
    conn: RocpdImportData,
    query: str,
    params: Union[Tuple, List] = (),
    export_format: Optional[str] = None,
    export_path: Optional[str] = None,
    dashboard_template_path: Optional[str] = None,
) -> Optional[str]:
    """
    Execute a SQLite query and print it to console.
    Then, if export_format is specified, write the results to a file.
    Returns the path to the exported file (or None if nothing was exported).

    Supported export_format values (case-insensitive):
        - "csv"
        - "html"
        - "md"   (markdown)
        - "pdf"
        - "dashboard"   (templated HTML dashboard)
        - "clipboard"

    If export_format == "dashboard", you may optionally pass a
    dashboard_template_path (a Jinja2 template file). If omitted,
    a built-in default template is used.
    """

    try:
        import pandas as pd

        conn = conn.connection if isinstance(conn, RocpdImportData) else conn

        # 1) Run the query via pandas
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            sys.stderr.write(f"No results found for query: {query}\n")
            sys.stderr.flush()
            return None

        if export_format == "console" or export_format is None:
            # 2) Print to console
            print(df.to_string(index=False))
            return None

        elif export_format == "clipboard":
            df.to_clipboard(excel=False)
            return None

        export_format = export_format.lower()
        ext = export_format
        export_path = export_path or f"query_output.{ext}"
        if not export_path.endswith(f".{ext}"):
            export_path = f"{export_path}.{ext}"
        export_path = os.path.abspath(libpyrocpd.format_path(export_path, "rocpd"))

        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        def write_export(content):
            with open(export_path, "w") as ofs:
                ofs.write(f"{content}\n")
                ofs.flush()

        # 3) Export based on format
        if export_format == "csv":
            df.to_csv(export_path, index=False)

        elif export_format == "html":
            write_export(df.to_html(index=False))

        elif export_format == "md":
            # pandas 1.0+ has to_markdown
            try:
                write_export(df.to_markdown(index=False))
            except AttributeError:
                # fallback: manually write markdown table
                _df_to_markdown_fallback(df, export_path)

        elif export_format == "pdf":
            _export_df_to_pdf(df, export_path)

        elif export_format == "dashboard":
            _export_dashboard(
                df, export_path=export_path, template_path=dashboard_template_path
            )

        elif export_format == "json":
            df.to_json(export_path, index=False, indent=2, orient="records")

        else:
            print(f"Unsupported export format: {export_format}")
            return None

        print(f"Exported to: {export_path}\n")
        return export_path

    except Exception as e:
        print(f"Error: {e}")
        return None


def _df_to_markdown_fallback(df, path: str):
    """
    Simple fallback if pandas.DataFrame.to_markdown(...) is unavailable.
    """
    headers = list(df.columns)
    with open(path, "w", encoding="utf-8") as f:
        # Header row
        f.write("| " + " | ".join(headers) + " |\n")
        # Separator
        f.write("|" + "|".join("---" for _ in headers) + "|\n")
        # Data rows
        for row in df.itertuples(index=False):
            line = "| " + " | ".join(str(v) for v in row) + " |\n"
            f.write(line)


def _export_df_to_pdf(df, path: str):
    """
    Render a DataFrame into a monospaced text table inside a PDF.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    x = 0.5 * inch
    y = height - 1 * inch
    row_height = 14

    c.setFont("Courier", 9)
    headers = list(df.columns)
    header_line = " | ".join(headers)
    c.drawString(x, y, header_line)
    y -= row_height
    c.drawString(x, y, "-" * len(header_line))
    y -= row_height

    for _, row in df.iterrows():
        row_line = " | ".join(str(v) for v in row)
        # Clip at ~160 characters so it doesn’t overflow the page width
        c.drawString(x, y, row_line[:160])
        y -= row_height
        if y < 1 * inch:
            c.showPage()
            c.setFont("Courier", 9)
            y = height - 1 * inch

    c.save()


def _export_dashboard(df, export_path: str, template_path: Optional[str] = None):
    """
    Generate a templated HTML “dashboard” from df. If template_path is None,
    use a built-in template. Otherwise, load the Jinja2 template from that path.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    # 1) Prepare Jinja2 environment
    if template_path:
        # User provided a .html (Jinja2) file
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(template_path)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(os.path.basename(template_path))
    else:
        # Built-in default template
        builtin_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <title>Dashboard Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #aaa; padding: 8px; text-align: left; }
                th { background-color: #f0f0f0; }
                tr:nth-child(even) { background-color: #fafafa; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <p><em>Generated on {{ timestamp }}</em></p>
            <div>
                {{ table_html | safe }}
            </div>
        </body>
        </html>
        """
        env = Environment(autoescape=select_autoescape(["html", "xml"]))
        template = env.from_string(builtin_html)

    # 2) Render template with context
    context = {
        "title": "SQLite Query Dashboard",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "table_html": df.to_html(index=False, classes="dashboard-table"),
    }
    rendered = template.render(**context)

    # 3) Write to export_path
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(rendered)


def zip_files(file_paths: List[str], zip_path: str) -> str:
    """
    Zip up one or more files into zip_path. Overwrites existing zip if present.
    Returns the path to the created zip.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.isfile(fp):
                zf.write(fp, arcname=os.path.basename(fp))
            else:
                raise FileNotFoundError(f"Cannot find file to zip: {fp}")
    print(f"Created ZIP archive: {zip_path}")
    return zip_path


def send_report_email(
    file_paths: List[str],
    to: Union[str, List[str]],
    sender: str,
    subject: str = "rocpd query Report",
    inline_preview: bool = False,
    smtp_server: str = "localhost",
    smtp_port: int = 25,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    zip_attachments: bool = False,
) -> None:
    """
    Send an email with one or more attachments, optionally zipped,
    and optionally with an inline preview (if the primary attachment is HTML).

    Args:
        file_paths: List of file paths to attach (each must exist).
        to: Recipient email address, or list of addresses.
        sender: Sender email address.
        subject: Subject line.
        inline_preview: If True, and one of the attachments is HTML, use that
                        HTML as the email body (and still attach files).
        smtp_server / smtp_port / smtp_user / smtp_password: SMTP credentials.
        zip_attachments: If True, bundle all file_paths into a single ZIP named
                         "<timestamp>_attachments.zip" and attach that ZIP only.
    """
    import smtplib
    from email.message import EmailMessage

    # 1) Validate that files exist
    for fp in file_paths:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Attachment not found: {fp}")

    # 2) If zip_attachments is True, zip everything into one archive
    actual_attachments: List[str]
    if zip_attachments:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = f"attachments_{timestamp}.zip"
        zip_files(file_paths, zip_path)
        actual_attachments = [zip_path]
    else:
        actual_attachments = file_paths.copy()

    # 3) Build the EmailMessage
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to) if isinstance(to, list) else to

    # 4) If inline_preview is True, look for the first HTML attachment,
    #    read its content, and set as an HTML alternative in the email body.
    if inline_preview:
        html_body_found = False
        for fp in actual_attachments:
            if fp.lower().endswith(".html"):
                with open(fp, "r", encoding="utf-8") as f:
                    html_content = f.read()
                msg.set_content(
                    "This email contains an inline HTML preview. If your mail client "
                    "doesn’t display HTML, see the attachment."
                )
                msg.add_alternative(html_content, subtype="html")
                html_body_found = True
                break
        if not html_body_found:
            # No HTML attachment found; create a simple text body
            msg.set_content("Please see attached report file(s).")

    else:
        # No inline preview desired; use a simple text body
        msg.set_content("Please see attached report file(s).")

    # 5) Attach each file (or the single ZIP)
    for fp in actual_attachments:
        with open(fp, "rb") as f:
            data = f.read()
        ctype = "application"
        subtype = "octet-stream"
        filename = os.path.basename(fp)
        msg.add_attachment(data, maintype=ctype, subtype=subtype, filename=filename)

    # 6) Connect to SMTP and send
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        if smtp_user and smtp_password:
            server.starttls()
            server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"Email sent to {msg['To']} with subject '{subject}'")


def add_args(parser):
    """Add query arguments"""

    query_options = parser.add_argument_group("Query Options")

    # Common arguments
    query_options.add_argument(
        "--query", required=True, help="SQL SELECT query to execute (enclose in quotes)."
    )

    query_options.add_argument(
        "--script",
        required=False,
        type=str,
        help="Input SQL script which should be read before query (e.g. defines views)",
    )

    query_options.add_argument(
        "--format",
        help="Export format",
        choices=("console", "csv", "html", "json", "md", "pdf", "dashboard", "clipboard"),
        type=str.lower,
    )

    email_options = parser.add_argument_group("Query Email Options")

    # Email options (optional)
    email_options.add_argument(
        "--email-to", help="Recipient email address (or comma-separated list)."
    )
    email_options.add_argument(
        "--email-from", help="Sender email address (required if --email-to is used)."
    )
    email_options.add_argument(
        "--email-subject",
        default="SQLite Query Report",
        help="Subject line for the email (default: %(default)s).",
    )
    email_options.add_argument(
        "--smtp-server",
        default="localhost",
        help="SMTP server hostname (default: %(default)s).",
    )
    email_options.add_argument(
        "--smtp-port",
        type=int,
        default=25,
        help="SMTP server port (default: %(default)d).",
    )
    email_options.add_argument("--smtp-user", help="SMTP login username (if required).")
    email_options.add_argument(
        "--smtp-password", help="SMTP login password (if required)."
    )
    email_options.add_argument(
        "--zip-attachments",
        action="store_true",
        help="Zip all attachments into a single .zip file before sending.",
    )
    email_options.add_argument(
        "--inline-preview",
        action="store_true",
        help="Embed HTML report as inline body if an HTML attachment is present.",
    )

    dashboard_options = parser.add_argument_group("Query Dashboard Options")

    dashboard_options.add_argument(
        "--template-path", help="Path to a Jinja2 HTML template for the dashboard"
    )

    return [
        "query",
        "script",
        "email_to",
        "email_from",
        "email_subject",
        "smtp_server",
        "smtp_port",
        "smtp_user",
        "smtp_password",
        "inline_preview",
        "zip_attachments",
        "format",
        "template_path",
    ]


def process_args(args, valid_args):
    # do not add any of the arguments to the output config dict
    ret = {}
    return ret


def execute(input, args, config=None, window_args=None, **kwargs):

    importData = RocpdImportData(input)

    apply_time_window(importData, **window_args)

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    if args.script:
        # read script and execute statements
        with open(args.script, "r") as ifs:
            for itr in ifs.read().split(";"):
                importData.execute(f"{itr}")

    # Prepare parameters for export
    query = args.query
    db = importData
    export_format = args.format
    export_path = os.path.join(config.output_path, config.output_file)

    # Dashboard-only extra
    dashboard_template = kwargs.get("template_path", None)

    # 1) Run and export
    exported_file = export_sqlite_query(
        db,
        query=query,
        params=(),
        export_format=export_format,
        export_path=export_path,
        dashboard_template_path=dashboard_template,
    )

    # 2) If --email-to was provided and we have a file, send it
    if args.email_to:
        if not args.email_from:
            raise ValueError("--email-from is required when --email-to is used.")
        if not exported_file:
            print("No file was exported; skipping email.")
            return

        recipients = [addr.strip() for addr in args.email_to.split(",")]
        send_report_email(
            file_paths=[exported_file],
            to=recipients,
            sender=args.email_from,
            subject=args.email_subject,
            inline_preview=args.inline_preview,
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
            smtp_user=args.smtp_user,
            smtp_password=args.smtp_password,
            zip_attachments=args.zip_attachments,
        )


def main(argv=None):
    import argparse
    from .time_window import add_args as add_args_time_window
    from .time_window import process_args as process_args_time_window
    from .output_config import add_args as add_args_output_config
    from .output_config import process_args as process_args_output_config
    from .output_config import add_generic_args, process_generic_args

    parser = argparse.ArgumentParser(
        description="Generate report for rocpd query", allow_abbrev=False
    )

    required_params = parser.add_argument_group("Required options")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    valid_out_config_args = add_args_output_config(parser)
    valid_generic_args = add_generic_args(parser)
    valid_time_window_args = add_args_time_window(parser)
    valid_query_args = add_args(parser)

    args = parser.parse_args(argv)

    out_cfg_args = process_args_output_config(args, valid_out_config_args)
    generic_out_cfg_args = process_generic_args(args, valid_generic_args)
    window_args = process_args_time_window(args, valid_time_window_args)
    query_args = process_args(args, valid_query_args)

    all_args = {
        **query_args,
        **out_cfg_args,
        **generic_out_cfg_args,
    }

    execute(
        args.input,
        args,
        window_args=window_args,
        **all_args,
    )


if __name__ == "__main__":
    main()
