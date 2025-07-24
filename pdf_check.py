import os
import pdfplumber

# Percorso della cartella dei PDF
pdf_dir = "Hackapizza Dataset/Menu"

# Trova tutti i PDF nella cartella e sottocartelle
all_pdfs = []
for root, dirs, files in os.walk(pdf_dir):
    for file in files:
        if file.lower().endswith(".pdf"):
            all_pdfs.append(os.path.join(root, file))

print(f"Trovati {len(all_pdfs)} PDF nella cartella '{pdf_dir}'.\n")

# Prova a caricare ogni PDF singolarmente con pdfplumber
for pdf_path in sorted(all_pdfs):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            # Prova a estrarre testo dalla prima pagina come test
            testo = pdf.pages[0].extract_text() if num_pages > 0 else None
        print(f"✅ LETTO: {pdf_path} (pagine: {num_pages})")
    except Exception as e:
        print(f"❌ ERRORE: {pdf_path}\n   Motivo: {e}\n") 