from flask import Flask, render_template, jsonify, request
import json
import os
import pickle
import numpy as np

app = Flask(__name__)

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
model_path = os.path.join(DATA_DIR, 'mudra_classifier.pkl')
encoder_path = os.path.join(DATA_DIR, 'label_encoder.pkl')

# Global cache
app_model = None
app_le = None

# Comprehensive dictionary of mudra meanings
MUDRA_MEANINGS = {
    'Alapadmam': 'Fully bloomed lotus or full moon',
    'Anjali': 'Salutation or greeting (Namaskar)',
    'Aralam': 'Bent or crooked (like a bird taking flight)',
    'Ardhachandran': 'Half moon, or spear',
    'Ardhapathaka': 'Half flag (bank of a river, a board)',
    'Berunda': 'Two-headed mythical bird',
    'Bramaram': 'Bee, or wings of a bee',
    'Chakra': 'Wheel or discus',
    'Chandrakala': 'Crescent moon, or a face',
    'Chaturam': 'Square, measure, or little quantity',
    'Garuda': 'Eagle (vehicle of Lord Vishnu)',
    'Hamsapaksha': 'Swan\'s wing, or arranging things',
    'Hamsasyam': 'Swan\'s beak, or tying a thread',
    'Kangulam': 'Tail, or bell, or betel nut',
    'Kapith': 'Elephant apple (Wood apple), or holding a weapon',
    'Kapotham': 'Dove, or respectful acceptance',
    'Karkatta': 'Crab, or blowing a conch',
    'Kartariswastika': 'Crossed scissors, or trees/branches',
    'Katakamukha': 'Opening of a bracelet (picking flowers/holding mirror)',
    'Katakavardhana': 'Link or chain (coronation, worship)',
    'Katrimukha': 'Scissors (corner of an eye, lightning)',
    'Khatva': 'Bed or palanquin',
    'Kilaka': 'Bond, affection, or a joke',
    'Kurma': 'Tortoise',
    'Matsya': 'Fish',
    'Mayura': 'Peacock, or wiping tears',
    'Mrigasirsha': 'Deer\'s head (holding an umbrella, or calling)',
    'Mukulam': 'Flower bud, or eating',
    'Mushti': 'Fist (grasping, steadfastness, fighting)',
    'Nagabandha': 'Coiled serpents',
    'Padmakosha': 'Lotus bud (fruit, circular shape)',
    'Pasha': 'Noose or rope (enmity, bondage)',
    'Pathaka': 'Flag (clouds, forest, forbidding things)',
    'Pushpaputa': 'Handful of flowers (offering)',
    'Sakata': 'Chariot or demon',
    'Samputa': 'Casket, or concealing things',
    'Sarpasirsha': 'Serpent\'s hood (snake, offering water)',
    'Shanka': 'Conch shell',
    'Shivalinga': 'Symbol of Lord Shiva',
    'Shukatundam': 'Parrot\'s beak (shooting an arrow, ferocity)',
    'Sikharam': 'Peak/Spire (holding a bow, or drinking)',
    'Simhamukham': 'Lion\'s face (hare, or preparation of medicine)',
    'Suchi': 'Needle (pointing, threatening, or "One")',
    'Swastikam': 'Crossed (crocodile, or blocking)',
    'Tamarachudam': 'Rooster (crane, crow, or camel)',
    'Tripathaka': 'Three parts of a flag (crown, tree, Indra)',
    'Trishulam': 'Trident (Lord Shiva\'s weapon)',
    'Varaha': 'Boar (incarnation of Lord Vishnu)'
}


def get_models():
    global app_model, app_le
    if not app_model and os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            app_model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            app_le = pickle.load(f)
    return app_model, app_le

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/library')
def library():
    return render_template('library.html', mudras=MUDRA_MEANINGS)

@app.route('/practice')
def practice():
    mudra_names = list(MUDRA_MEANINGS.keys())
    return render_template('practice.html', mudra_names=mudra_names)

def clean_mudra_name(raw_name):
    """ Cleans up names from dataset like 'Alapadmam(1)' -> 'Alapadmam' or 'Katakamukha_1' -> 'Katakamukha' """
    clean = raw_name.replace('(1)', '').strip()
    if '_' in clean:
        clean = clean.split('_')[0]
    return clean

@app.route('/api/predict', methods=['POST'])
def predict():
    mdl, lbl_enc = get_models()
    if not mdl or not lbl_enc:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    landmarks = data.get('landmarks')

    if not landmarks or len(landmarks) != 63:
        return jsonify({"error": "Invalid landmarks"}), 400

    # Predict mudra with confidence
    lm_arr = np.array([landmarks])
    pred_idx = mdl.predict(lm_arr)[0]
    raw_mudra = lbl_enc.inverse_transform([pred_idx])[0]
    
    clean_name = clean_mudra_name(raw_mudra)
    meaning = MUDRA_MEANINGS.get(clean_name, "Traditional hand gesture")

    # Get confidence from probabilities if supported
    confidence = 0
    if hasattr(mdl, 'predict_proba'):
        proba = mdl.predict_proba(lm_arr)[0]
        confidence = int(round(max(proba) * 100))

    return jsonify({
        "predicted": clean_name,
        "meaning": meaning,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
