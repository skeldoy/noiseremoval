import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk

# Directories
dirs = {
    "training": "../data/inference",
    "autoencoders": "../data/autoencoders",
    "n2n": "../data/n2n",
    "cyclegan": "../data/cyclegan"
}

# Get list of image files from the 'training' directory
file_list = sorted(os.listdir(dirs["training"]))

# Initialize current index
current_index = 0

def load_images(index):
    """ Load images from each directory based on the current index. """
    images = {}
    for name, path in dirs.items():
        img_path = os.path.join(path, file_list[index])
        img = Image.open(img_path)
        img = img.resize((screen_width // 2, screen_height // 2), Image.LANCZOS)
        
        # Add overlay text
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        text = name
        textbbox = draw.textbbox((0, 0), text, font=font)
        textwidth = textbbox[2] - textbbox[0]
        textheight = textbbox[3] - textbbox[1]
        width, height = img.size
        # Position the text at the bottom right
        x = width - textwidth - 10
        y = height - textheight - 10
        draw.text((x, y), text, font=font, fill="white")
        
        images[name] = ImageTk.PhotoImage(img)
    return images

def update_images(index):
    """ Update the images displayed in the GUI. """
    global current_index
    current_index = index
    images = load_images(current_index)
    for name, label in image_labels.items():
        label.config(image=images[name])
        label.image = images[name]

def next_image(event):
    """ Display the next image. """
    global current_index
    if current_index < len(file_list) - 1:
        update_images(current_index + 1)

def previous_image(event):
    """ Display the previous image. """
    global current_index
    if current_index > 0:
        update_images(current_index - 1)

# Create the main window
root = tk.Tk()
root.title("Image Comparison")

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to fit the screen
root.geometry(f"{screen_width}x{screen_height}")

# Create a frame for the images
frame = tk.Frame(root)
frame.pack()

# Create labels to hold the images
image_labels = {}
for i, name in enumerate(dirs.keys()):
    label = tk.Label(frame)
    row = i // 2
    col = i % 2
    label.grid(row=row, column=col)
    image_labels[name] = label

# Load and display the initial images
update_images(current_index)

# Bind the arrow keys to the functions
root.bind("<Right>", next_image)
root.bind("<Left>", previous_image)

# Start the GUI event loop
root.mainloop()

