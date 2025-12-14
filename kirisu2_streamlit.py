import io
import os
import zipfile
from copy import deepcopy

import streamlit as st

from tdspectrum import TDSpectrum


# Set page config
st.set_page_config(page_title="Kirisu 2",
                   page_icon="logo.png",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Title and Description
st.title("Kirisu 2")
st.markdown("*Simple editor for stopped-flow and other spectra*")

# Sidebar
st.sidebar.header("1. Upload Spectra")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (GLB, CSV, BK3A, ZIP)",
    help="Upload one or multiple spectra files or a ZIP containing them.",
    accept_multiple_files=True)
input_format = st.sidebar.selectbox(
    "Input Format (Optional)", ["", "glb", "csv", "bk3a"],
    index=0,
    help=
    "Leave empty to auto-detect from extension. If the format is not recognized, GLB will be assumed."
)

st.sidebar.header("2. Process Spectra")

plot_2d_flag = st.sidebar.toggle("Plot 2D Spectra", value=True)
plot_3d_flag = st.sidebar.toggle("Plot 3D Spectra", value=False)

# Smoothing
st.sidebar.subheader("Smoothing")
smooth_method = st.sidebar.segmented_control(
    "Smoothing Method",
    options=["no", "gaussian", "sma", "median"],
    default="no",
    help="""**no** - Do not apply smoothing \\
            **gaussian** - Gaussian Filter \\
            **sma** - Simple Moving Average \\
            **median** - Median Filter""")
if smooth_method is None:
    smooth_method = "no"
smooth_level = st.sidebar.slider("Smoothing Level", 0.2, 3.0, 1.0, 0.1)

# Zeroing
st.sidebar.subheader(
    "Zeroing",
    help="Modify the spectra to make all times zero at a specific wavelength.")
do_zero = st.sidebar.checkbox("Zero at Wavelength", value=True)
lambda_zero = st.sidebar.number_input("Wavelength (nm)", value=1000.0)

# Trimming
st.sidebar.subheader(
    "Trimming",
    help=
    "Trim the spectra to only include up to a minimum and maximum time and wavelength."
)
col1, col2 = st.sidebar.columns(2)
with col1:
    time_min = st.number_input("Min Time", value=0.0)
    lambda_min = st.number_input("Min Lambda", value=0.0)
with col2:
    time_max = st.number_input("Max Time", value=1000.0)
    lambda_max = st.number_input("Max Lambda", value=1000.0)

# Save Section
st.sidebar.header(
    "3. Save",
    help=
    "Choose the output format. If no format is selected, the original file extension will be used. Files can also be bundled into a single ZIP file."
)
output_format = st.sidebar.selectbox("Output Format",
                                     ["", "glb", "csv", "bk3a"],
                                     index=0)
bundle_zip = st.sidebar.toggle("Bundle as ZIP", value=False)

# Authorship and License
st.sidebar.markdown("---")
st.sidebar.markdown('''
by Sergio Boneta Martínez

based on the original *kirisu*
by Jose Ramón Peregrina and Jorge Estrada

GPLv3 (C) 2022-2025 \\
@ Universidad de Zaragoza
''')

# Main Area
if not uploaded_files:
    st.info("Please upload spectra files in the sidebar to begin.")
else:
    spectra = []

    # Read Files
    with st.spinner("Reading files..."):
        for uploaded_file in uploaded_files:
            # Zip
            is_zip = False
            try:
                is_zip = zipfile.is_zipfile(uploaded_file)
            except:
                pass
            uploaded_file.seek(0)

            if is_zip:
                with zipfile.ZipFile(uploaded_file) as zf:
                    for zi in zf.infolist():
                        if zi.file_size == 0:
                            continue
                        with zf.open(zi) as f:
                            try:
                                filestr = f.read().decode('cp1252')
                                s = TDSpectrum()
                                s.read(zi.filename,
                                       filestr,
                                       format=input_format)
                                spectra.append(s)
                            except Exception as e:
                                st.error(
                                    f"Error reading {zi.filename} inside zip: {e}"
                                )
            else:
                # Single File
                try:
                    filestr = uploaded_file.read().decode('cp1252')
                    s = TDSpectrum()
                    s.read(uploaded_file.name, filestr, format=input_format)
                    spectra.append(s)
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {e}")

    if not spectra:
        st.warning("No valid spectra loaded.")
        st.stop()

    st.success(f"Successfully loaded {len(spectra)} spectra.")

    # Process Spectra
    processed_spectra = []
    with st.spinner("Processing spectra..."):
        for spectrum in spectra:
            s = deepcopy(spectrum)

            # Zero
            if do_zero:
                s.zero(lambda_zero)

            # Smooth
            if smooth_method != "no":
                s.smooth(smooth_method, smooth_level)

            # Trim
            s.trim([time_min, time_max], [lambda_min, lambda_max])

            processed_spectra.append(s)

    # Create tabs for each spectrum
        for s in processed_spectra:
            with st.expander(f"File: {s.filename}", expanded=True):
                st.text(str(s))
                tab1, tab2, tab3 = st.tabs([
                    "Time vs Absorbance", "Wavelength vs Absorbance",
                    "Wavelength/Absorbance vs Time"
                ])
                with tab1:
                    if plot_2d_flag:
                        st.plotly_chart(s.plot("2d-times", "plotly"),
                                        use_container_width=True)
                with tab2:
                    if plot_2d_flag:
                        st.plotly_chart(s.plot("2d-lambdas", "plotly"),
                                        use_container_width=True)
                with tab3:
                    if plot_3d_flag:
                        st.plotly_chart(s.plot("3d", "plotly"), use_container_width=True)

    # Save Section
    st.markdown("---")
    st.subheader("Save Processed Spectra")

    if bundle_zip:
        # Create Zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for s in processed_spectra:
                try:
                    data_str = s.formatted_string(format=output_format)
                    # Determine filename
                    fname = s.filename_trim
                    # If output format is specified, ensure extension matches
                    if output_format:
                        base = os.path.splitext(fname)[0]
                        fname = f"{base}.{output_format}"
                    elif not os.path.splitext(fname)[1]:
                        fname = f"{fname}.glb"

                    zf.writestr(fname, data_str)
                except Exception as e:
                    st.error(f"Error preparing {s.filename} for save: {e}")

        st.download_button(label="Save All (ZIP)",
                           data=zip_buffer.getvalue(),
                           file_name="spectra_kirisu2.zip",
                           mime="application/zip")
    else:
        # Individual Saves
        cols = st.columns(min(3, len(processed_spectra)))
        for i, s in enumerate(processed_spectra):
            col = cols[i % 3]
            try:
                data_str = s.formatted_string(format=output_format)
                fname = s.filename_trim
                if output_format:
                    base = os.path.splitext(fname)[0]
                    fname = f"{base}.{output_format}"
                elif not os.path.splitext(fname)[1]:
                    fname = f"{fname}.glb"

                col.download_button(label=f"Download {fname}",
                                    data=data_str,
                                    file_name=fname,
                                    mime="text/plain",
                                    key=f"dl_{i}")
            except Exception as e:
                col.error(f"Error: {e}")
