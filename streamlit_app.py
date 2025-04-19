# streamlit_app.py
import os
import io
import json
import psycopg2
import timm
import streamlit as st
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import torch, torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download

# ─── 1. CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()

def get_config(section: str, key: str, env_var: str, default=None):
    """
    Try Streamlit secrets[section][key], 
    then os.getenv(env_var), 
    finally default (or error).
    """
    # 1) Streamlit Cloud secrets
    try:
        return st.secrets[section][key]
    except Exception:
        pass

    # 2) Local .env
    val = os.getenv(env_var, default)
    if val is None:
        st.error(f"⚠️ Missing config for '{section}.{key}' or env var '{env_var}'")
        st.stop()
    return val

# ─── 1) Database URL ──────────────────────────────────────────────────────────
DATABASE_URL = get_config("database", "url", "DATABASE_URL")

# ─── 2) HuggingFace model info ────────────────────────────────────────────────
HF_REPO_ID  = get_config("model", "repo_id",  "MODEL_REPO_ID")
HF_FILENAME = get_config("model", "filename", "MODEL_FILENAME", default="checkpoint_best_2.pt")
HF_TOKEN    = get_config("model", "hf_token",  "HF_TOKEN", default=None)

# ─── 3) Download & cache your checkpoint ─────────────────────────────────────
@st.cache_resource
def fetch_model():
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        use_auth_token=HF_TOKEN
    )

MODEL_PATH = fetch_model()
MAPPING_PATH = "machinelearning/species_mapping.json"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()
if not os.path.exists(MAPPING_PATH):
    st.error(f"Mapping file not found at {MAPPING_PATH}")
    st.stop()

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2. LOAD SPECIES MAPPING ──────────────────────────────────────────────────
with open(MAPPING_PATH, "r") as f:
    species_names = json.load(f)  # e.g. ["Acanthurus triostegus", "Amphiprion ocellaris", ...]

 # ─── 3. LOAD MODEL ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with st.spinner("⏳ Loading model…"):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        net = timm.create_model("efficientnet_b0",
                                pretrained=False,
                                num_classes=len(species_names))
        net.load_state_dict(state_dict)
        return net.to(DEVICE).eval()

model = load_model()

# ─── 4. PREPROCESS TRANSFORMS ──────────────────────────────────────────────────
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─── 5. DB HELPERS ─────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL)

def init_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS catches (
                id          SERIAL PRIMARY KEY,
                img_data    BYTEA       NOT NULL,
                species     TEXT        NOT NULL,
                lure        TEXT,
                size        REAL,
                location    TEXT,
                condition   TEXT,
                favourite   BOOLEAN,
                created_at  TIMESTAMP   DEFAULT NOW()
            );
            """)
    # no need to commit() in `with` block

init_table()

def insert_catch(img_bytes, species, lure, size, loc, cond, fav):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO catches (img_data, species, lure, size, location, condition, favourite)
            VALUES (%s,%s,%s,%s,%s,%s,%s);
            """, (psycopg2.Binary(img_bytes), species, lure, size, loc, cond, fav))

def fetch_past():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, img_data, species, lure, size, location, condition, favourite, created_at
                  FROM catches
                 ORDER BY created_at DESC
            """)
            return cur.fetchall()

def update_catch(catch_id, species, lure, size, loc, cond, fav):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE catches
                   SET species=%s, lure=%s, size=%s, location=%s, condition=%s, favourite=%s
                 WHERE id=%s
            """, (species, lure, size, loc, cond, fav, catch_id))

# ─── 6. PREDICTION ─────────────────────────────────────────────────────────────
def predict_species(image: Image.Image):
    x = val_tfms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
    return species_names[idx], probs[0, idx].item()

# ─── 7. STREAMLIT UI ──────────────────────────────────────────────────────────
st.title("🐳 Whaley The Fish Finder 🎣")

# --- Upload & Inputs
uploaded = st.file_uploader("Upload fish image", type=["jpg","png","jpeg"])
lure    = st.text_input("Lure / Bait used")
size    = st.number_input("Size (cm)", min_value=0.0, step=0.1)
loc     = st.text_input("Location")
cond    = st.selectbox("Condition", ["first light","mid day", "last light", "night"])
fav     = st.checkbox("Favourite catch?")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your upload", use_column_width=True)
    if st.button("Classify & Save"):
        # 1) predict
        species, conf = predict_species(img)
        st.success(f"**Predicted:** {species} ({conf*100:.1f}%)")

        # 2) save
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        insert_catch(img_bytes.getvalue(), species, lure, size, loc, cond, fav)
        st.info("✅ Saved to your catch log.")

# --- Display Past Catches (with edit forms) -----------------
st.header("📜 Your Past Catches")
rows = fetch_past()
if not rows:
    st.write("No entries yet.")
else:
    for (cid, img_data, species, lure, size, location, condition, favourite, ts) in rows:
        with st.expander(f"#{species} from {ts:%Y‑%m‑%d %H:%M}", expanded=False):
            # show the saved image
            buf = BytesIO(img_data.tobytes())
            img = Image.open(buf)
            st.image(img, width=500)

            # all fields in a form so they update together
            with st.form(f"edit_form_{cid}", clear_on_submit=False):
                new_species = st.text_input("Species", value=species)
                new_lure    = st.text_input("Lure / Bait", value=lure)
                new_size    = st.number_input("Size (cm)", value=size, step=0.1)
                new_loc     = st.text_input("Location", value=location)
                new_cond    = st.selectbox("Condition",
                                           ["first light","mid day","last light","night"],
                                           index=["first light","mid day","last light","night"].index(condition))
                new_fav     = st.checkbox("Favourite", value=favourite)

                submitted = st.form_submit_button("Update this catch")
                if submitted:
                    update_catch(cid,
                                 new_species,
                                 new_lure,
                                 new_size,
                                 new_loc,
                                 new_cond,
                                 new_fav)
                    st.success("Catch updated!")
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        st.info("🔄 Please refresh the page to see your updated catch.")