import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import pydeck as pdk
# Modell laden
model = joblib.load('models/Onehot_log_modell.pkl')

airport_coords = {
    "ABQ": {"lat": 35.0402, "lon": -106.6090},
    "ACK": {"lat": 41.2531, "lon": -70.0602},
    "AGS": {"lat": 33.3699, "lon": -81.9643},
    "ALB": {"lat": 42.7481, "lon": -73.8029},
    "ANC": {"lat": 61.1743, "lon": -149.9962},
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "AUS": {"lat": 30.1975, "lon": -97.6664},
    "AVL": {"lat": 35.4362, "lon": -82.5418},
    "AVP": {"lat": 41.3385, "lon": -75.7243},
    "BDL": {"lat": 41.9389, "lon": -72.6832},
    "BGM": {"lat": 42.2087, "lon": -75.9794},
    "BGR": {"lat": 44.8070, "lon": -68.8281},
    "BHM": {"lat": 33.5629, "lon": -86.7535},
    "BNA": {"lat": 36.1263, "lon": -86.6774},
    "BOS": {"lat": 42.3656, "lon": -71.0096},
    "BQN": {"lat": 18.4949, "lon": -67.1294},
    "BTV": {"lat": 44.4719, "lon": -73.1532},
    "BUF": {"lat": 42.9405, "lon": -78.7322},
    "BUR": {"lat": 34.2007, "lon": -118.3587},
    "BWI": {"lat": 39.1754, "lon": -76.6684},
    "BZN": {"lat": 45.7776, "lon": -111.1605},
    "CAE": {"lat": 33.9388, "lon": -81.1195},
    "CHO": {"lat": 38.1386, "lon": -78.4529},
    "CHS": {"lat": 32.8986, "lon": -80.0405},
    "CLE": {"lat": 41.4125, "lon": -81.8498},
    "CLT": {"lat": 35.2140, "lon": -80.9431},
    "CMH": {"lat": 39.9980, "lon": -82.8919},
    "CVG": {"lat": 39.0558, "lon": -84.6675},
    "DAL": {"lat": 32.8471, "lon": -96.8517},
    "DAY": {"lat": 39.9024, "lon": -84.2194},
    "DCA": {"lat": 38.8512, "lon": -77.0402},
    "DEN": {"lat": 39.8561, "lon": -104.6737},
    "DFW": {"lat": 32.8998, "lon": -97.0403},
    "DSM": {"lat": 41.5341, "lon": -93.6631},
    "DTW": {"lat": 42.2124, "lon": -83.3534},
    "EGE": {"lat": 39.6426, "lon": -106.9177},
    "EYW": {"lat": 24.5561, "lon": -81.7596},
    "FLL": {"lat": 26.0726, "lon": -80.1527},
    "GRR": {"lat": 42.8808, "lon": -85.5228},
    "GSO": {"lat": 36.0800, "lon": -79.9410},
    "GSP": {"lat": 34.8956, "lon": -82.2189},
    "HDN": {"lat": 40.4812, "lon": -106.8166},
    "HHH": {"lat": 32.2244, "lon": -80.6975},
    "HNL": {"lat": 21.3187, "lon": -157.9224},
    "HOU": {"lat": 29.6454, "lon": -95.2789},
    "HYA": {"lat": 41.6693, "lon": -70.2803},
    "IAD": {"lat": 38.9531, "lon": -77.4565},
    "IAH": {"lat": 29.9902, "lon": -95.3368},
    "ILM": {"lat": 34.2706, "lon": -77.9026},
    "IND": {"lat": 39.7169, "lon": -86.2956},
    "ITH": {"lat": 42.4910, "lon": -76.4584},
    "JAC": {"lat": 43.6076, "lon": -110.7376},
    "JAX": {"lat": 30.4941, "lon": -81.6879},
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "LAS": {"lat": 36.0840, "lon": -115.1537},
    "LAX": {"lat": 33.9416, "lon": -118.4085},
    "LGA": {"lat": 40.7769, "lon": -73.8740},
    "LIT": {"lat": 34.7294, "lon": -92.2243},
    "MCI": {"lat": 39.2976, "lon": -94.7139},
    "MCO": {"lat": 28.4312, "lon": -81.3081},
    "MDT": {"lat": 40.1935, "lon": -76.7634},
    "MDW": {"lat": 41.7868, "lon": -87.7522},
    "MEM": {"lat": 35.0446, "lon": -89.9810},
    "MHT": {"lat": 42.9345, "lon": -71.4371},
    "MIA": {"lat": 25.7959, "lon": -80.2870},
    "MKE": {"lat": 42.9474, "lon": -87.8966},
    "MSN": {"lat": 43.1399, "lon": -89.3375},
    "MSP": {"lat": 44.8848, "lon": -93.2223},
    "MSY": {"lat": 29.9934, "lon": -90.2580},
    "MTJ": {"lat": 38.5098, "lon": -107.8942},
    "MVY": {"lat": 41.3930, "lon": -70.6143},
    "MYR": {"lat": 33.6827, "lon": -78.9288},
    "OAK": {"lat": 37.7126, "lon": -122.2197},
    "OGG": {"lat": 20.8986, "lon": -156.4305},
    "OKC": {"lat": 35.3931, "lon": -97.6007},
    "OMA": {"lat": 41.2996, "lon": -95.8979},
    "ONT": {"lat": 34.0556, "lon": -117.6012},
    "ORD": {"lat": 41.9742, "lon": -87.9073},
    "ORF": {"lat": 36.8946, "lon": -76.2012},
    "ORH": {"lat": 42.2675, "lon": -71.8757},
    "PBI": {"lat": 26.6832, "lon": -80.0956},
    "PDX": {"lat": 45.5887, "lon": -122.5975},
    "PHL": {"lat": 39.8744, "lon": -75.2424},
    "PHX": {"lat": 33.4343, "lon": -112.0116},
    "PIT": {"lat": 40.4915, "lon": -80.2329},
    "PNS": {"lat": 30.4736, "lon": -87.1866},
    "PSE": {"lat": 18.0083, "lon": -66.5630},
    "PSP": {"lat": 33.8297, "lon": -116.5062},
    "PVD": {"lat": 41.7326, "lon": -71.4204},
    "PWM": {"lat": 43.6462, "lon": -70.3093},
    "RDU": {"lat": 35.8801, "lon": -78.7880},
    "RIC": {"lat": 37.5052, "lon": -77.3197},
    "RNO": {"lat": 39.4991, "lon": -119.7680},
    "ROA": {"lat": 37.3255, "lon": -79.9754},
    "ROC": {"lat": 43.1200, "lon": -77.6724},
    "RSW": {"lat": 26.5362, "lon": -81.7552},
    "SAN": {"lat": 32.7338, "lon": -117.1933},
    "SAT": {"lat": 29.4268, "lon": -98.4866},
    "SAV": {"lat": 32.1276, "lon": -81.2023},
    "SBN": {"lat": 41.7087, "lon": -86.3185},
    "SCE": {"lat": 40.8493, "lon": -77.8489},
    "SDF": {"lat": 38.1744, "lon": -85.7360},
    "SEA": {"lat": 47.4502, "lon": -122.3088},
    "SFO": {"lat": 37.6213, "lon": -122.3790},
    "SJC": {"lat": 37.3626, "lon": -121.9290},
    "SJU": {"lat": 18.4394, "lon": -66.0018},
    "SLC": {"lat": 40.7899, "lon": -111.9791},
    "SMF": {"lat": 38.6951, "lon": -121.5900},
    "SNA": {"lat": 33.6757, "lon": -117.8682},
    "SRQ": {"lat": 27.3954, "lon": -82.5544},
    "STL": {"lat": 38.7487, "lon": -90.3700},
    "STT": {"lat": 18.3361, "lon": -64.9750},
    "SYR": {"lat": 43.1112, "lon": -76.1063},
    "TPA": {"lat": 27.9755, "lon": -82.5332},
    "TUL": {"lat": 36.2003, "lon": -95.8882},
    "TVC": {"lat": 44.7414, "lon": -85.5822},
    "TYS": {"lat": 35.8110, "lon": -83.9940},
    "VPS": {"lat": 30.4833, "lon": -86.5254},
    "XNA": {"lat": 36.2819, "lon": -94.3068},
    "EWR": {"lat": 40.6895, "lon": -74.1745}
}



# Encoder laden (wir speichern die Encoder mit, um sie später zu verwenden)
encoder_origin = joblib.load('models/Encoder/encoder_origin.pkl')
encoder_dest = joblib.load('models/Encoder/encoder_dest.pkl')

# Titel der Web-App
st.title('Vorhersage der Ankunftsverspätung')

# Eingabefelder für den Benutzer
# Eingabefelder mit optionaler Validierung
sched_dep_time = st.number_input("Geplante Abflugzeit (hhmm, z.B. 0040 für 00:40)", min_value=0, max_value=2359, value=0)
if sched_dep_time != 0:  # Prüfen, ob der Nutzer die Zeit geändert hat
    if sched_dep_time // 100 < 24 and sched_dep_time % 100 < 60:
        st.success("✔️ Zeit korrekt eingegeben.")
    else:
        st.error("❌ Ungültige Zeit, bitte hhmm-Format beachten.")

# Tag des Monats
day = st.number_input("Tag des Monats (1-31)", min_value=0, max_value=31, step=1, value=0)
if day == 0:  # Überprüfen, ob der Standardwert 0 noch ausgewählt ist
    st.error("❗ Bitte einen gültigen Tag (1-31) eingeben.")
else:
    st.success("✔️ Tag korrekt eingegeben.")

# Monat
month = st.number_input("Monat (1-12)", min_value=0, max_value=12, step=1, value=0)
if month == 0:  # Überprüfen, ob der Standardwert 0 noch ausgewählt ist
    st.error("❗ Bitte einen gültigen Monat (1-12) eingeben.")
else:
    st.success("✔️ Monat korrekt eingegeben.")



origin = st.selectbox('Abflughafen (origin)', ['JFK', 'LGA', 'EWR'])  # Beispielhafte Flughäfen
dest = st.selectbox('Zielflughafen (dest)', encoder_dest.categories_[0])  # Verwende alle Ziel-Flughäfen aus den Encodern

if st.button('Vorhersage'):
    # Erstelle DataFrames für 'origin' und 'dest', um Spaltennamen zu beibehalten
    origin_df = pd.DataFrame([origin], columns=['origin'])
    dest_df = pd.DataFrame([dest], columns=['dest'])

    # One-Hot-Encoding der Eingaben für 'origin' und 'dest'
    origin_encoded = encoder_origin.transform(origin_df)
    dest_encoded = encoder_dest.transform(dest_df)

    # Eingaben kombinieren: Numerische Daten und die One-Hot-codierten Daten
    input_data = np.concatenate([[[sched_dep_time, day, month]], origin_encoded, dest_encoded], axis=1)

    # Spaltennamen für das DataFrame erstellen
    origin_columns = encoder_origin.categories_[0]  # Kategorien für 'origin'
    dest_columns = encoder_dest.categories_[0]      # Kategorien für 'dest'

    # Erstelle einen DataFrame für die Eingabedaten, der mit den Spaltennamen übereinstimmt
    input_df = pd.DataFrame(input_data, columns=['sched_dep_time', 'day', 'month'] + list(origin_columns) + list(dest_columns))

    # Vorhersage durchführen
    prediction_proba = model.predict_proba(input_df)

    # Wahrscheinlichkeit der Klasse "Verspätung" (Klasse 1)
    delay_probability = prediction_proba[0][1]

    # Ergebnis anzeigen
    if delay_probability >= 0.5:
        st.write(f"Vorhersage: Verspätung erwartet!")
        st.write(f"Die Wahrscheinlichkeit für eine Verspätung beträgt: {delay_probability:.2f}")
    else:
        st.write(f"Vorhersage: Pünktliche Ankunft erwartet!")
        st.write(f"Die Wahrscheinlichkeit für eine Verspätung beträgt: {delay_probability:.2f}")



# Flughäfen-Koordinaten extrahieren
    origin_coords = airport_coords.get(origin, {"lat": 0, "lon": 0})
    dest_coords = airport_coords.get(dest, {"lat": 0, "lon": 0})

    # Karte anzeigen
    st.write("Karte der ausgewählten Flughäfen:")
    map_data = pd.DataFrame([
        {"name": "Abflug: " + origin, "lat": origin_coords["lat"], "lon": origin_coords["lon"]},
        {"name": "Ziel: " + dest, "lat": dest_coords["lat"], "lon": dest_coords["lon"]}
    ])

    # Pydeck-Karte erstellen
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[lon, lat]",
        get_color="[200, 30, 0, 160]",
        get_radius=50000,
    )
    view_state = pdk.ViewState(latitude=map_data["lat"].mean(), longitude=map_data["lon"].mean(), zoom=5, pitch=0)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})
    st.pydeck_chart(deck)