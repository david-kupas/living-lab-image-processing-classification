import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from io import BytesIO

import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch


# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# App title
st.title("Living Lab: Interaktív Képosztályozó Eszköz")
st.write("Fedezd fel a Streamlit képességeit egy képelőfeldolgozó és képosztályozó alkalmazásban!")

# Image upload
uploaded_file = st.file_uploader("Tölts fel egy képet:", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Feltöltött kép", use_container_width=True)


# Interactive preprocessing
if uploaded_file:
    st.sidebar.header("Képelőfeldolgozás")

    # Store the original image only once
    if "original_image" not in st.session_state:
        st.session_state.original_image = Image.open(uploaded_file)

    # Initialize session state variables only if they do not exist
    if "grayscale" not in st.session_state:
        st.session_state["grayscale"] = False
    if "color_filter" not in st.session_state:
        st.session_state["color_filter"] = "Nincs"
    if "crop_size" not in st.session_state:
        st.session_state["crop_size"] = "Nincs"
    if "invert_colors" not in st.session_state:
        st.session_state["invert_colors"] = False
    if "filters" not in st.session_state:
        st.session_state["filters"] = []
    if "blur_radius" not in st.session_state:
        st.session_state["blur_radius"] = 2
    if "resize_width" not in st.session_state:
        st.session_state["resize_width"] = st.session_state.original_image.width
    if "resize_height" not in st.session_state:
        st.session_state["resize_height"] = st.session_state.original_image.height

    # Define a function to reset all session state variables
    def reset_all():
        st.session_state["grayscale"] = False
        st.session_state["color_filter"] = "Nincs"
        st.session_state["crop_size"] = "Nincs"
        st.session_state["invert_colors"] = False
        st.session_state["filters"] = []
        st.session_state["blur_radius"] = 2
        st.session_state["resize_width"] = st.session_state.original_image.width
        st.session_state["resize_height"] = st.session_state.original_image.height

    # Add a reset button
    if st.sidebar.button("Visszaállítás alapértelmezettre"):
        reset_all()

    # Load the image to be processed
    image = st.session_state.original_image.copy()

    # Grayscale
    grayscale = st.sidebar.checkbox("Átváltás szürkeárnyalatos képre", key="grayscale")
    if grayscale:
        image = image.convert("L")

    # Color Filter
    color_filter = st.sidebar.radio(
        "Színszűrő alkalmazása:",
        ["Nincs", "Piros", "Zöld", "Kék"],
        index=0,
        key="color_filter"
    )
    if color_filter == "Piros":
        image = image.point(lambda p: p if p < 128 else 255)
    elif color_filter == "Zöld":
        image = image.point(lambda p: p if p < 128 else 128)
    elif color_filter == "Kék":
        image = image.point(lambda p: p if p < 128 else 64)

    # Crop
    crop_size = st.sidebar.selectbox(
        "Kép kivágása:",
        ["Nincs", "Négyzet (300x300)", "Szélesvásznú (500x300)"],
        index=0,
        key="crop_size"
    )
    if crop_size == "Négyzet (300x300)":
        image = image.crop((0, 0, 300, 300))
    elif crop_size == "Szélesvásznú (500x300)":
        image = image.crop((0, 0, 500, 300))

    # Invert Colors
    invert_colors = st.sidebar.checkbox("Színek megfordítása", key="invert_colors")
    if invert_colors:
        image = ImageOps.invert(image)

    # Multiselect Filters
    filters = st.sidebar.multiselect(
        "Szűrők alkalmazása:",
        ["Éldetektálás", "Homályosítás", "Élesítés", "Dombornyomás", "Fényerő beállítása", "Kontraszt beállítása",
         "Poszterizálás"],
        key="filters"
    )

    if "Éldetektálás" in filters:
        image = image.filter(ImageFilter.FIND_EDGES)
    if "Homályosítás" in filters:
        blur_radius = st.sidebar.slider(
            "Homályosítás mértéke", 1, 10, 2, key="blur_radius"
        )
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    if "Élesítés" in filters:
        image = image.filter(ImageFilter.SHARPEN)
    if "Dombornyomás" in filters:
        image = image.filter(ImageFilter.EMBOSS)
    if "Fényerő beállítása" in filters:
        brightness_factor = st.sidebar.slider(
            "Fényerő mértéke", 0.1, 2.0, 1.0, key="brightness_factor"
        )
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
    if "Kontraszt beállítása" in filters:
        contrast_factor = st.sidebar.slider(
            "Kontraszt mértéke", 0.1, 2.0, 1.0, key="contrast_factor"
        )
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
    if "Poszterizálás" in filters:
        posterize_bits = st.sidebar.slider(
            "Poszterizálás (színek száma)", 1, 8, 8, key="posterize_bits"
        )
        image = ImageOps.posterize(image, posterize_bits)

    # Resize with Number Input
    width = st.sidebar.number_input(
        "Új szélesség (pixel):",
        min_value=10,
        key="resize_width"
    )
    height = st.sidebar.number_input(
        "Új magasság (pixel):",
        min_value=10,
        key="resize_height"
    )
    image = image.resize((int(width), int(height)))

    # Display preprocessed image
    st.image(image, caption="Előfeldolgozott kép", use_container_width=True)

    # Download image
    if st.button("Előfeldolgozott kép mentése"):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        st.download_button(
            label="Letöltés",
            data=buffered.getvalue(),
            file_name="preprocessed_image.png",
            mime="image/png"
        )

    # Classification Functionality
    st.sidebar.header("Képosztályozás")

    # Load pre-trained ResNet152 model
    from torchvision.models import resnet152

    model = resnet152(pretrained=True)
    model.eval()  # Set the model to evaluation mode


    # Define ImageNet class labels loader
    @st.cache_data
    def load_imagenet_classes():
        import requests
        response = requests.get(
            "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")
        return response.json()


    imagenet_classes = load_imagenet_classes()

    # Add a classify button
    if st.button("Kép osztályozása"):
        with st.spinner("A kép osztályozása folyamatban..."):
            # Preprocess the image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Simulate longer processing time
            import time

            time.sleep(2)  # Add artificial delay to showcase the spinner

            # Run the model and get predictions
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_class = torch.topk(probabilities, 5)  # Top-5 predictions

            # Prepare results
            results = [(imagenet_classes[top_class[i].item()], top_prob[i].item() * 100) for i in range(5)]

        # Display results in a fancy format
        st.success("Osztályozás kész!")
        st.markdown("### **Eredmények**")

        # Highlight the top-1 prediction
        top_label, top_confidence = results[0]
        st.markdown(f"#### 🏆 **Legvalószínűbb osztály**: `{top_label}`")
        st.markdown(f"#### 🔥 **Bizonyosság**: `{top_confidence:.2f}%`")

        # Display Top-5 predictions in a table
        st.markdown("### **Top-5 Predikciók**")
        st.table(
            {
                "Osztály": [res[0] for res in results],
                "Bizonyosság (%)": [f"{res[1]:.2f}" for res in results],
            }
        )






