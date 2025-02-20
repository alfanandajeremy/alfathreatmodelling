import streamlit as st

# Title of the app
st.title("ALFA THREAT MODELLING ")

# Input for system description
system_description = st.text_area("Deskripsi Aplikasi")

# Button to submit the threat model
if st.button("Submit Threat Model"):
    st.success("Threat model submitted successfully!")
    st.write("System Description:", system_description)
    st.write("Assets:", assets.split(","))
    st.write("Potential Threats:", threats.split(","))
    st.write("Vulnerabilities:", vulnerabilities.split(","))
    st.write("Mitigations:", mitigations.split(","))

# Run the app using: streamlit run your_script.py