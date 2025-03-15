import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from io import StringIO
import random

# Page Config muss als erstes kommen!
st.set_page_config(page_title="MNER Tagger App", page_icon="📝")

# Titel & Beschreibung
st.markdown("# MNER Tagger App")   
st.markdown("This app tags texts in English, French, German, and Italian using the WikiNeural model.")

# Modell und Tokenizer laden
model_name = r"C:\Users\Julia\Documents\CAS NLP\Modul 4 Transformers\Model\finetuned_wikineural"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Label-Mapping
label_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}

# Tabs definieren
tab1, tab2 = st.tabs(["📂 Upload a File", "✍ Enter Text"])

# 📂 **Tab 1: Datei-Upload**
with tab1:
    st.write("Upload a file with raw text. The output will be a file formatted as: 'token mner_tag'.")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # Datei in einen String umwandeln
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()

        # Tokenisierung
        tokens = tokenizer(string_data.split(), is_split_into_words=True, return_tensors="pt")

        # Modellinferenz
        with torch.no_grad():
            outputs = model(**tokens)

        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())

        # Ausgabe formatieren
        tagged_output = "\n".join(f"{token} {label_map[pred]}" for token, pred in zip(tokens_list, predictions))

        # Zufälligen Dateinamen generieren
        file_name = f"tagged_text_{random.randint(10000, 100000)}.txt"

        # Datei-Download-Button
        st.download_button("Download Tagged File", tagged_output, file_name, "text/plain")

        # Vorschau anzeigen
        st.text_area("Preview:", tagged_output[:500], height=200)

# ✍ **Tab 2: Manuelle Texteingabe**
with tab2:
    text = st.text_input("Insert a text to get the POS tags for it")

    if text:
        # Tokenisierung mit Rückgabe der Wortgrenzen
        tokens = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt")

        # Modell-Inferenz
        with torch.no_grad():
            outputs = model(**tokens)

        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())

        # HTML-Ausgabe vorbereiten
        html_results = ""
        for token, pred in zip(tokens_list, predictions):
            tag = label_map.get(pred, "O")  # Standardwert "O", falls kein Mapping existiert
            html_results += f"<span style='color:red;'>{token} </span><span style='color:blue;'>[{tag}]</span> "

        # Ergebnis anzeigen
        st.markdown(html_results, unsafe_allow_html=True)


