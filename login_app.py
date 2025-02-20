import streamlit as st

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="Login - Alfa Threat Modelling",
    page_icon="shield"  # Ganti dengan nama file ikon Anda
)

# Data login
USERS = {
    "jeremy": "jeremyalfananda",
}

# Fungsi autentikasi
def authenticate(username, password):
    return USERS.get(username) == password

# Halaman login
def login_app():
    st.title("Login Page")

    # Form login
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    # Proses login
    if login_button:
        if authenticate(username, password):
            st.success("Login berhasil! ")
            st.write("Selamat datang,", username)
            # Menampilkan tautan yang dapat diklik untuk redirect
            st.markdown("[Klik di sini untuk lanjut ke halaman utama](http://localhost:8501/)")
        else:
            st.error("Username atau password salah")

# Menjalankan halaman login
login_app()