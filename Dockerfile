FROM python:3.11-slim-buster

WORKDIR /app

# Copy the processing script
COPY process_all_pdfs.py .

# Run the script
CMD ["python", "process_pdfs.py"] 