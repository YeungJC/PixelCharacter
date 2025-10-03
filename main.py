import io

import streamlit as st

from users import Users
from db import ImageDatabase


from models import Decoder
import torch
from PIL import Image
import numpy as np

# Load
decoder = Decoder(shape_before_flattening=torch.Size([128, 8, 8]))
# Attempt to load trained model
try:
    decoder.load_state_dict(torch.load("decoder.pt"))
    device = torch.device("cpu")
except:
    # Display error message if model not found
    st.error("Could not load pixel generation model")

# Title page
st.title("Character Creator")

def logout():
    # Logout of user session
    st.session_state.pop("user_id")

def signup(username, password, email): 
    # check that username password and email meets valid
    passed_validation = Users().validate(username, password, email) 
    # if meets validation, signup user (create db entry)
    if passed_validation:
        user_id = Users().signup(username, password, email) #calls the signup function in the class Users passing in three parameters: username,password and email
        # Add user_id and username to session which we will reuse to view / save images
        st.session_state["user_id"] = user_id
        st.session_state["username"] = username
    return None


def login(username, password): 
    # login user with username password
    user_id = Users().login(username, password) 
    if user_id:
        # Add user_id and username to session which we will reuse to view / save images
        st.session_state["user_id"] = user_id 
        st.session_state["username"] = username
    return None


def user_authentication(option): 
    # displaying different options to user depending on login / singup
    if option == "signup":
    # User needs username, password and email to signup
        signup_username = st.text_input("username", key="signup_username") 
        signup_password = st.text_input("password", key="signup_password", type="password")
        signup_email = st.text_input("email", key="signup_email")
    # If user presses button, run signup function with signup_username, signup_password, signup_email passed to function
        st.button(
            "Signup",
            on_click=signup,
            args=(signup_username, signup_password, signup_email),
        )
    else:
    # User needs username, password to login
        username = st.text_input("username", key="login_username")
        password = st.text_input("password", key="login_password", type="password")
        # if user presses button, run login function passing in username and password
        st.button("Login", on_click=login, args=(username, password))
    return None

def create_image_from_decoder(latent_vector):
    # Use decoder model to convert latent vector to image
    generated_image = decoder(latent_vector.to(device))
    generated_image = generated_image.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    # Scale decoder output to 255 bitmap
    generated_image = generated_image.astype(np.float32) / np.max(generated_image)
    generated_image = (generated_image * 255).astype(np.uint8)

    # Convert numpy array to png image
    generated_image = Image.fromarray(generated_image)
    with io.BytesIO() as output:
        generated_image.save(output, format="PNG")
        generated_image = output.getvalue()
    return generated_image


def create_image():
    variation = 2
    # also related to not training the model - no grad means no gradient
    with torch.no_grad():
        # Generate random vector
        random_latent_vector = torch.randn(1, 40) * variation
        random_image = create_image_from_decoder(random_latent_vector)

    # Save latest generated image and latent vector to global variable
    # which will be used if user decides to save image
    st.session_state["image"] = random_image
    st.session_state["latent_vector"] = random_latent_vector.tolist()[0]
    return None


def save_image(label):
    # Saves randomly generated image to database
    # insert record to database
    id = ImageDatabase().create_image_details_record(st.session_state["user_id"], label)
    # Create sql query for 40 columns 
    latent_vectors = ", ".join([f"latent_vector_{i}" for i in range(40)])
    placeholders = "?," * 40
    args1 = [id, st.session_state["image"]]
    args2 = [*args1, *st.session_state["latent_vector"]]
    # Insert binary of image and latent vector to db
    ImageDatabase().insert_raw_image(latent_vectors, placeholders, args2)


def generate_page():
    image_label = None
    # When use clicks generate image button, randomly generate image
    st.button("Generate image", on_click=create_image)
    # When user first loads page, save page is disabled as no image has been generated
    disable_save = True
    if "image" in st.session_state:
        # If user has generated an image, display
        st.image(
            st.session_state["image"], width=256, caption="Randomly generated image"
        )
        # Give user option to save image with an image name
        image_label = st.text_input("Image name")
        disable_save = False
    st.button(
        "Save image", on_click=save_image, disabled=disable_save, args=[image_label]
    )
    return None


def delete_image(img_id):
    # Delete image record
    ImageDatabase().delete_image_id_from_image_details(img_id)
    # Delete raw image binary along with latent vectors
    ImageDatabase().delete_image_id_from_image_binary(img_id)


def saved_page():
    # subheading
    st.subheader("View generated images")
    # Get all images saved corresponding to user_oid
    records = ImageDatabase().get_image_details(st.session_state["user_id"])
    # get all image ids
    ids = [i[1] for i in records]
    # get all image names
    names = [i[2] for i in records]
    # get binary for all images
    imgs = [i[3] for i in records]
    # Creating a display name from id and name, as names are not unique
    img_names = [f"{i}:{j}" for i, j in zip(ids, names)]
    # Display all user images. st.image accepts an array of images and captions
    st.image(imgs, img_names, width=256)
    # Select box for user to choose any images to delete
    delete_img = st.selectbox("Delete image, please select id", options=ids, index=None)
    # Disable delete button until user has selected an id
    disable_button = delete_img is None
    # Delete an image
    st.button(
        "Delete", on_click=delete_image, args=[delete_img], disabled=disable_button
    )


def blend_images(img_id1, img_id2, blend_ratio):
    # Display names created as image_id:image_name
    # Splitting gives us the image id
    img_id1 = img_id1.split(":")[0]
    img_id2 = img_id2.split(":")[0]
    # Get the image binary and latent vectors
    records = ImageDatabase().get_image_pairs(img_id1, img_id2)
    # Formatting latent vectors to pytorch
    img_vector1 = torch.reshape(torch.tensor(records[0]), (1, 40))
    img_vector2 = torch.reshape(torch.tensor(records[1]), (1, 40))

    # remember, we are interpolating in the latent space, not mixing the actual images. That's why this is so cool
    random_latent_vector_3 = ((1 - blend_ratio) * img_vector1) + (
        blend_ratio * img_vector2
    )
    with torch.no_grad():
        # Use blended latent vectors to generate an image
        random_image = create_image_from_decoder(random_latent_vector_3)
    # Save latest generated image and latent vector to global variable
    # which will be used if user decides to save image
    st.session_state["image_blend"] = random_image
    st.session_state["latent_vector_blend"] = random_latent_vector_3.tolist()[0]


def save_blend_image(label):
    # Create record in image_details db
    id = ImageDatabase().create_image_details_record(st.session_state["user_id"], label)
    # Generate latent vector sql columns
    latent_vectors = ", ".join([f"latent_vector_{i}" for i in range(40)])
    placeholders = "?," * 40
    args1 = [id, st.session_state["image_blend"]]
    args2 = [*args1, *st.session_state["latent_vector_blend"]]
    # Insert image binary and latent vectors to db
    ImageDatabase().insert_raw_image(latent_vectors, placeholders, args2)


def blend_page():
    # Get all saved images for user
    records = ImageDatabase().get_image_details(st.session_state["user_id"])
    # Get all img ids
    ids = [i[1] for i in records]
    # Get all img names
    names = [i[2] for i in records]
    # Get all img binary
    imgs = [i[3] for i in records]
    # Construct display name
    img_names = [f"{i}:{j}" for i, j in zip(ids, names)]
    # User select image 1
    img1 = st.selectbox("Image 1", options=img_names, index=None)
    # User select image 2. We remove whatever user selected from 1 from potential list
    img2 = st.selectbox(
        "Image 2", options=[i for i in img_names if i != img1], index=None
    )
    # Use streamlit columns to show image side by side
    img1_show, img2_show = st.columns(2)
    if img1 is not None:
        # Display image 1 if user has selected something
        with img1_show:
            st.image(imgs[img_names.index(img1)], width=256)
    if img2 is not None:
        # Display image 2 if user has selected something
        with img2_show:
            st.image(imgs[img_names.index(img2)], width=256)
    # Disable blend button until user has selected an image for 1 and 2
    disabled_blend = True
    if (img1 is not None) & (img2 is not None):
        disabled_blend = False
    # Slider for user to select blend ratio, 0 complete copy of image 1, 1 complete copy of image 2
    blend_ratio = st.slider("Blend ratio", 0.0, 1.0, value=0.5, step=0.1)
    st.button("Blend", on_click=blend_images, args=(img1, img2, blend_ratio), disabled= disabled_blend)
    # Blend images by averaging latent vectors and creating image
    if "image_blend" in st.session_state:
        # Allow user to save image with a name
        st.image(st.session_state["image_blend"], width=256)
        image_label = st.text_input("Image name")
        st.button("Save blended image", on_click=save_blend_image, args=[image_label])

def update_email_address(email):
    # Update user email address record in db with new email
    # Will check validation and return error if not meeting standards
    Users().update_email(email, st.session_state["user_id"])
    return None

def update_password(password):
    # Update user passwrod record in db with new password
    # Will check validation and return error if not meeting standards
    Users().update_password(password, st.session_state["user_id"])
    return None

def settings_page():
    # Get email address of user to display
    user_details = Users().get_user_details(st.session_state["user_id"])
    st.subheader("Email address")
    st.write(user_details[1])
    # Text input for user to input new email
    new_email = st.text_input("New email address")
    # Button to validate and update email
    st.button("Update email address", on_click=update_email_address, args=[new_email])
    st.subheader("Password")
    # Button to validate and update password
    new_password = st.text_input("New password")
    st.button("Update password", on_click=update_password, args=[new_password])

def main():
    # If user has logged in / signed up successfully, user_id will be in st.session state
    # So we can display the main app
    # Else direct them to login / sign up
    if "user_id" not in st.session_state:
        # Radio button to switch between sign  up and logib
        option = st.radio("", ["signup", "login"])
        user_authentication(option)

    if "user_id" in st.session_state:
        # 4 pages in user app, generate, view saved, blend and edit settings
        page = st.sidebar.radio(
            "Page Navigation", ["Generate images", "Saved images", "Blend Images", "User settings"]
        )
        # lOGOUT button to change user
        st.sidebar.button("Logout", on_click=logout)
        # Display username to user
        st.code(f"Hello {st.session_state['username']}")
        # Switch statement to change pages
        if page == "Generate images":
            generate_page()
        elif page == "Saved images":
            saved_page()
        elif page == "Blend Images":
            blend_page()
        else:
            settings_page()

# Run main function
main()
