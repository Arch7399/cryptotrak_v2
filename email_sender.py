import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()


def send_email_alert(currencies, recipient_emails):
    sender_email = os.getenv("SENDER_EMAIL")
    password = os.getenv("PASSWORD")
    subject = "Cryptocurrency Alert"

    html_content = """
    <html>
    <head>
    <title>Cryptocurrency Alert</title>
    <style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f0f8ff;
    }
    .container {
        max-width: 700px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 {
        color: #333;
        font-size: 1.5rem;
        text-align: center;
    }
    p {
        margin: 0;
        padding: 5px;
        color: #666;
    }
    ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    li {
        display: inline-block;
        margin: 0 10px 10px 0;
        padding: 5px 10px;
        background-color: #e0ebef;
        border-radius: 5px;
    }
    </style>
    </head>
    <body>
    <div class="container">
    <h1>Cryptocurrency Alert</h1>
    <p>The following currencies are performing well:</p>
    <ul>
    """
    for currency in currencies:
        currency_url = currency.replace(" ", "-")
        url = f"https://coinmarketcap.com/currencies/{currency_url}/"
        html_content += f'<li><a href="{url}">{currency}</a></li>\n'

    html_content += """   
    </ul>
    </div>
    </body>
    </html>
    """

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(recipient_emails)
    message["Subject"] = subject

    html_part = MIMEText(html_content, "html")
    message.attach(html_part)

    with smtplib.SMTP("smtp.gmail.com", port=587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient_emails, message.as_string())
