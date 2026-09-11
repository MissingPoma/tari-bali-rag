import json
from flask import Flask, render_template, request
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from symspellpy import SymSpell, Verbosity
import google.generativeai as genai

# Konfigurasi API Gemini dengan API key Anda
GEMINI_API_KEY = "AIzaSyCEv92gTRSe3jzoD66yPrp9oRoOP7fE1gQ"
genai.configure(api_key=GEMINI_API_KEY)

# Inisialisasi model Gemini
model = genai.GenerativeModel('gemini-1.5-flash')

# Download NLTK data (hanya untuk tokenisasi)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Load data dari file JSON
with open('taribali4.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    dances = data['Tarian']

# Inisialisasi SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load kamus dari id_dict.json
with open('id_dict.json', 'r', encoding='utf-8') as f:
    dict_data = json.load(f)
    for entry in dict_data['dictionary']:
        sym_spell.create_dictionary_entry(entry['word'], entry['frequency'])

# Tambahkan kata-kata dari data tarian ke kamus SymSpell
for dance in dances:
    nama_tari = dance['nama_tari'].lower()
    if nama_tari.startswith("tari "):
        nama_spesifik = nama_tari.replace("tari ", "", 1)
        sym_spell.create_dictionary_entry("tari", 2000)
        sym_spell.create_dictionary_entry(nama_spesifik.replace(" ", ""), 1000)
        sym_spell.create_dictionary_entry(nama_spesifik, 1000)
        for word in nama_spesifik.split():
            sym_spell.create_dictionary_entry(word, 500)
    else:
        nama_spesifik = nama_tari
        sym_spell.create_dictionary_entry(nama_spesifik.replace(" ", ""), 1000)
        sym_spell.create_dictionary_entry(nama_spesifik, 1000)
        for word in nama_spesifik.split():
            sym_spell.create_dictionary_entry(word, 500)
    sym_spell.create_dictionary_entry(nama_tari, 1000)
    
    sym_spell.create_dictionary_entry(dance['jenis_tari'].lower(), 800)
    
    for word in dance['deskripsi'].lower().split():
        if word.isalnum():
            sym_spell.create_dictionary_entry(word, 500)
    
    for word in dance['makna'].lower().split():
        if word.isalnum():
            sym_spell.create_dictionary_entry(word, 500)
    
    pencipta = dance.get('pencipta', '').lower()
    if pencipta and pencipta != '-':
        sym_spell.create_dictionary_entry(pencipta, 1000)
        for word in pencipta.split():
            if word.isalnum():
                sym_spell.create_dictionary_entry(word, 500)

# Preprocessing: Gabungkan semua teks relevan dari setiap tarian untuk pencarian
documents = []
for dance in dances:
    text = f"{dance['nama_tari']} {dance['jenis_tari']} {dance['deskripsi']} {dance['isi_artikel']} {dance['makna']}"
    documents.append(text)

# Inisialisasi stopwords dan stemmer dari Sastrawi
factory_stop = StopWordRemoverFactory()
stop_words = set(factory_stop.get_stop_words())

factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

# Fungsi untuk menangani karakter berulang
def normalize_repeated_chars(word):
    return re.sub(r'(.)\1+', r'\1', word)

# Fungsi untuk membersihkan teks menggunakan Sastrawi
def preprocess_text(text):
    text = normalize_repeated_chars(text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess semua dokumen
processed_docs = [preprocess_text(doc) for doc in documents]

# Inisialisasi TF-IDF Vectorizer dengan n-gram
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(processed_docs)

def correct_typo(query):
    original_words = query.split()
    query_lower = query.lower()
    for dance in dances:
        nama_tari = dance['nama_tari'].lower()
        nama_spesifik = nama_tari.replace("tari ", "", 1) if nama_tari.startswith("tari ") else nama_tari
        if query_lower == nama_spesifik or query_lower.replace(" ", "") == nama_spesifik.replace(" ", ""):
            return dance['nama_tari'].lower()

    def match_case(original, corrected):
        if not original or not corrected:
            return corrected
        if len(original) != len(corrected):
            if original.isupper():
                return corrected.upper()
            elif original[0].isupper():
                return corrected.capitalize()
            else:
                return corrected
        result = []
        for i in range(len(original)):
            if i < len(corrected):
                if original[i].isupper():
                    result.append(corrected[i].upper())
                else:
                    result.append(corrected[i].lower())
            else:
                break
        return ''.join(result)

    corrected_query = []
    for i, word in enumerate(original_words):
        normalized_word = normalize_repeated_chars(word.lower())
        suggestions = sym_spell.lookup(normalized_word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_word = suggestions[0].term
            if corrected_word == word.lower():
                corrected_query.append(word)
            else:
                corrected_query.append(match_case(word, corrected_word))
        else:
            corrected_query.append(word)

    return ' '.join(corrected_query)

def is_pencipta_query(query):
    pencipta_keywords = ["pencipta", "siapa", "diciptakan", "membuat", "cipta"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in pencipta_keywords)

def is_location_query(query):
    location_keywords = ["dari", "berasal", "asal"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in location_keywords)

def extract_location(query):
    query_lower = query.lower()
    words = query_lower.split()
    for i, word in enumerate(words):
        if word in ["dari", "berasal", "asal"] and i + 1 < len(words):
            return words[i + 1]
    return None

def find_relevant_sentence(query, dance):
    texts = [
        dance.get('deskripsi', ''),
        dance.get('isi_artikel', ''),
        dance.get('makna', '')
    ]
    
    if dance.get('elemen_tari'):
        elemen_tari = dance['elemen_tari']
        gerakan = elemen_tari.get('gerakan', '')
        properti = elemen_tari.get('properti', '')
        busana = elemen_tari.get('busana', '')
        
        if isinstance(gerakan, list):
            gerakan = ' '.join(gerakan)
        elif isinstance(gerakan, dict):
            gerakan = ' '.join(f"{k}: {v}" for k, v in gerakan.items() if isinstance(v, str))
        elif not isinstance(gerakan, str):
            gerakan = str(gerakan)
            
        if isinstance(properti, list):
            properti = ' '.join(properti)
        elif isinstance(properti, dict):
            properti = ' '.join(f"{k}: {v}" for k, v in properti.items() if isinstance(v, str))
        elif not isinstance(properti, str):
            properti = str(properti)
            
        if isinstance(busana, list):
            busana = ' '.join(busana)
        elif isinstance(busana, dict):
            busana = ' '.join(f"{k}: {v}" for k, v in busana.items() if isinstance(v, str))
        elif not isinstance(busana, str):
            busana = str(busana)
        
        texts.append(gerakan)
        texts.append(properti)
        texts.append(busana)
    
    texts = [text if isinstance(text, str) else str(text) for text in texts]
    full_text = ' '.join(texts)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return None
    
    processed_query = preprocess_text(query)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_query] + processed_sentences)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        ranked_indices = similarities.argsort()[::-1]
        
        top_sentences = []
        for idx in ranked_indices[:3]:
            if similarities[idx] >= 0.1:
                top_sentences.append(sentences[idx])
        
        if not top_sentences:
            return None
        
        return top_sentences
    except ValueError:
        return None

def create_filtered_context(relevant_dances):
    # Buat konteks hanya dari tarian yang relevan berdasarkan TF-IDF
    filtered_context = "Berikut adalah data tarian Bali yang relevan dengan query:\n"
    for dance in relevant_dances:
        elemen_tari_text = ""
        if dance.get('elemen_tari'):
            elemen_tari = dance['elemen_tari']
            gerakan = elemen_tari.get('gerakan', '')
            properti = elemen_tari.get('properti', '')
            busana = elemen_tari.get('busana', '')
            elemen_tari_text = f"Gerakan: {gerakan}, Properti: {properti}, Busana: {busana}"
        filtered_context += f"- Nama Tarian: {dance['nama_tari']}, Jenis Tari: {dance['jenis_tari']}, Pencipta: {dance.get('pencipta', '-')}, Deskripsi: {dance['deskripsi']}, Makna: {dance['makna']}, Isi Artikel: {dance['isi_artikel']}, Elemen Tari: {elemen_tari_text}\n"
    return filtered_context

def query_gemini(query, relevant_dances):
    context = create_filtered_context([dance for dance, score, _ in relevant_dances])
    
    try:
        if is_location_query(query):
            location = extract_location(query)
            prompt = f"{context}\n\nJawab pertanyaan berikut secara langsung tanpa preambel atau kalimat pengantar seperti 'Berdasarkan data yang diberikan': {query}. Jika informasi asal daerah tidak ada di field 'asal', cari di dalam 'Deskripsi', 'Isi Artikel', atau 'Elemen Tari' untuk menentukan apakah tarian ini berasal dari {location}."
        else:
            prompt = f"{context}\n\nJawab pertanyaan berikut secara langsung tanpa preambel atau kalimat pengantar seperti 'Berdasarkan data yang diberikan': {query}."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error accessing Gemini API: {str(e)}"

def search_dance(query):
    corrected_query = correct_typo(query)
    matching_dances = []
    
    # Cek apakah query mencari asal daerah
    if is_location_query(query):
        location = extract_location(query)
        if location:
            location_lower = location.lower()
            for dance in dances:
                # Cari di field asal jika ada
                if dance.get('asal', '').lower() == location_lower:
                    highlighted_info = f"Ber出身 dari {location}."
                    relevant_sentences = find_relevant_sentence(query, dance)
                    if relevant_sentences:
                        highlighted_info += " " + " ".join(relevant_sentences)
                    matching_dances.append((dance, 1.0, highlighted_info))
                    continue
                
                # Cari di deskripsi, isi artikel, dan elemen tari
                combined_text = f"{dance.get('deskripsi', '')} {dance.get('isi_artikel', '')} {dance.get('elemen_tari', {}).get('gerakan', '')} {dance.get('elemen_tari', {}).get('properti', '')} {dance.get('elemen_tari', {}).get('busana', '')}".lower()
                if location_lower in combined_text:
                    highlighted_info = f""
                    relevant_sentences = find_relevant_sentence(query, dance)
                    if relevant_sentences:
                        highlighted_info += " " + " ".join(relevant_sentences)
                    matching_dances.append((dance, 0.9, highlighted_info))
    
    if matching_dances:
        return matching_dances, corrected_query

    # Pencocokan langsung dengan nama tarian
    for i, dance in enumerate(dances):
        nama_tari = dance['nama_tari'].lower()
        nama_spesifik = nama_tari.replace("tari ", "", 1) if nama_tari.startswith("tari ") else nama_tari
        if (corrected_query.lower() == nama_spesifik or 
            corrected_query.lower().replace(" ", "") == nama_spesifik.replace(" ", "") or
            corrected_query.lower() == nama_tari):
            highlighted_info = ""
            if is_pencipta_query(query):
                pencipta = dance.get('pencipta', '-')
                if pencipta != '-':
                    highlighted_info += f"Pencipta: {pencipta}. "
            
            relevant_sentences = find_relevant_sentence(query, dance)
            if relevant_sentences:
                highlighted_info += " ".join(relevant_sentences)
            
            return [(dance, 1.0, highlighted_info)], corrected_query

    # Pencocokan langsung dengan nama pencipta
    for dance in dances:
        pencipta = dance.get('pencipta', '').lower()
        if pencipta and pencipta != '-' and corrected_query.lower() == pencipta:
            highlighted_info = f"Pencipta: {pencipta}. "
            relevant_sentences = find_relevant_sentence(query, dance)
            if relevant_sentences:
                highlighted_info += " ".join(relevant_sentences)
            matching_dances.append((dance, 1.0, highlighted_info))
    
    if matching_dances:
        return matching_dances, corrected_query

    # Pencarian TF-IDF
    processed_query = preprocess_text(corrected_query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]
    
    top_indices = ranked_indices[:3] if len(ranked_indices) >= 3 else ranked_indices
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append((dances[idx], similarities[idx]))

    highlighted_results = []
    for dance, score in results:
        highlighted_info = ""
        if is_pencipta_query(query):
            pencipta = dance.get('pencipta', '-')
            if pencipta != '-':
                highlighted_info += f"{pencipta}. "
        
        relevant_sentences = find_relevant_sentence(query, dance)
        if relevant_sentences:
            highlighted_info += " ".join(relevant_sentences)
        
        highlighted_results.append((dance, score, highlighted_info))
    
    return highlighted_results, corrected_query

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ''
    corrected_query = ''
    gemini_result = ''
    if request.method == 'POST':
        query = request.form['query']
        results, corrected_query = search_dance(query)
        gemini_result = query_gemini(query, results if results else [])
    return render_template('index3.html', gemini_result=gemini_result, results=results, query=query, corrected_query=corrected_query)

if __name__ == '__main__':
    app.run(debug=True)