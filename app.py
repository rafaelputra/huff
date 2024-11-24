from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import heapq
from collections import defaultdict
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class HuffmanNode:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(frequencies):
    heap = [HuffmanNode(value, freq) for value, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_codes(node, current_code="", codes={}):
    if node is None:
        return

    if node.value is not None:
        codes[node.value] = current_code
        return

    generate_codes(node.left, current_code + "0", codes)
    generate_codes(node.right, current_code + "1", codes)

def calculate_channel_frequencies(image):
    frequencies = {
        "R": defaultdict(int),
        "G": defaultdict(int),
        "B": defaultdict(int)
    }
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r, g, b = image[y, x]
            frequencies["R"][r] += 1
            frequencies["G"][g] += 1
            frequencies["B"][b] += 1
    return frequencies

def encode_color_image(image, codes):
    encoded_data = {
        "R": "".join([codes["R"][pixel[0]] for pixel in image.reshape(-1, 3)]),
        "G": "".join([codes["G"][pixel[1]] for pixel in image.reshape(-1, 3)]),
        "B": "".join([codes["B"][pixel[2]] for pixel in image.reshape(-1, 3)]),
    }
    return encoded_data

def decode_color_image(encoded_data, codes, image_shape):
    reverse_codes = {
        "R": {v: k for k, v in codes["R"].items()},
        "G": {v: k for k, v in codes["G"].items()},
        "B": {v: k for k, v in codes["B"].items()},
    }

    decoded_channels = {}
    for channel in ["R", "G", "B"]:
        current_code = ""
        decoded_channel = []
        for bit in encoded_data[channel]:
            current_code += bit
            if current_code in reverse_codes[channel]:
                decoded_channel.append(reverse_codes[channel][current_code])
                current_code = ""
        decoded_channels[channel] = np.array(decoded_channel, dtype=np.uint8)

    decoded_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    decoded_image[:, :, 0] = decoded_channels["R"].reshape(image_shape[:2])
    decoded_image[:, :, 1] = decoded_channels["G"].reshape(image_shape[:2])
    decoded_image[:, :, 2] = decoded_channels["B"].reshape(image_shape[:2])
    return decoded_image

def load_color_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)

def get_image_base64(image):
    buffered = BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str

import json

# Add this function inside your Flask app code
def tree_to_dict(node):
    if node is None:
        return None
    if isinstance(node.value, int):  # Assuming values are integers or None
        return {"name": str(node.value)}
    return {
        "name": str(node.value) if node.value is not None else "",
        "children": [tree_to_dict(node.left), tree_to_dict(node.right)],
    }


huffman_tree = None  # Add this at the top of your app, outside any function

@app.route('/', methods=['GET', 'POST'])
def home():
    global huffman_tree  # Use the global variable to store the tree
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image = load_color_image(file_path)
            frequencies = calculate_channel_frequencies(image)

            codes = {}
            for channel in ["R", "G", "B"]:
                huffman_tree = build_huffman_tree(frequencies[channel])  # Save the tree
                codes[channel] = {}
                generate_codes(huffman_tree, "", codes[channel])

            encoded_data = encode_color_image(image, codes)
            decoded_image = decode_color_image(encoded_data, codes, image.shape)

            original_size = image.size * 8  # 8 bits per pixel for RGB
            compressed_size = sum(len(encoded_data[channel]) for channel in ["R", "G", "B"])
            compression_ratio = original_size / compressed_size

            original_image_base64 = get_image_base64(image)
            decompressed_image_base64 = get_image_base64(decoded_image)

            return render_template('index.html', 
                                  original_image=original_image_base64,
                                  decompressed_image=decompressed_image_base64,
                                  compression_ratio=compression_ratio)

    return render_template('index.html')

@app.route('/get_tree_data', methods=['GET'])
def get_tree_data():
    global huffman_tree  # Use the globally stored tree
    if huffman_tree is None:
        return {"error": "Huffman tree not available"}, 400

    tree_dict = tree_to_dict(huffman_tree)
    return tree_dict  # Automatically converts to JSON


@app.route('/tree', methods=['GET'])
def visualize_tree():
    # Generate the dictionary representation of the tree
    tree_dict = tree_to_dict(huffman_tree)  # Assuming huffman_tree is defined

    # Convert the dictionary to JSON format
    tree_json = json.dumps(tree_dict, indent=4)

    # Save the JSON to a file
    with open('huffman_tree.json', 'w') as json_file:
        json_file.write(tree_json)
        # Render the tree visualization page
    return render_template('tree.html')


if __name__ == '__main__':
    app.run(debug=True, port=6969)
