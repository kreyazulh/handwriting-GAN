import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generator class (same as training script)
class Generator(nn.Module):
    def __init__(self, nz, num_classes, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Input: noise + class label
        input_dim = nz + num_classes
        
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            # Second layer
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            # Third layer
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            # Output layer
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Concatenate noise and labels
        x = torch.cat([noise, label_embedding], dim=1)
        # Generate image
        img = self.model(x)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

# Initialize model parameters
nz = 100
num_classes = 10
image_size = 28
device = torch.device('cpu')  # Use CPU for deployment

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    generator = Generator(nz, num_classes, image_size)
    
    try:
        # Load the trained weights
        generator.load_state_dict(torch.load('generator_final.pth', map_location=device))
        generator.eval()
        return generator
    except FileNotFoundError:
        st.error("Model file 'generator_final.pth' not found. Please train the model first.")
        return None

def generate_digit_images(generator, digit, num_images=5):
    """Generate images for a specific digit"""
    with torch.no_grad():
        # Create random noise
        noise = torch.randn(num_images, nz)
        # Create labels for the specific digit
        labels = torch.full((num_images,), digit, dtype=torch.long)
        
        # Generate images
        fake_images = generator(noise, labels)
        
        # Denormalize images from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # Convert to numpy and reshape
        images = fake_images.squeeze().numpy()
        
        return images

def numpy_to_pil(img_array):
    """Convert numpy array to PIL Image"""
    # Ensure values are in [0, 255] range
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

# Main app
def main():
    st.title("🔢 Handwritten Digit Image Generator")
    st.markdown("Generate synthetic MNIST-like handwritten digits using a trained Conditional GAN")
    
    # Load model
    generator = load_model()
    
    if generator is None:
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Generation Controls")
        
        # Digit selection
        selected_digit = st.selectbox(
            "Choose a digit to generate (0-9):",
            options=list(range(10)),
            index=2
        )
        
        # Generation button
        generate_button = st.button("🎲 Generate Images", type="primary")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses a Conditional GAN trained on the MNIST dataset to generate 
        handwritten digit images. Each generation creates 5 unique variations 
        of the selected digit.
        """)
    
    # Main content area
    if generate_button:
        with st.spinner(f"Generating images for digit {selected_digit}..."):
            # Generate images
            generated_images = generate_digit_images(generator, selected_digit, 5)
            
            st.success(f"✅ Generated 5 images for digit {selected_digit}")
            
            # Display images
            st.subheader(f"Generated Images of Digit {selected_digit}")
            
            # Create 5 columns for the images
            cols = st.columns(5)
            
            for i, img_array in enumerate(generated_images):
                with cols[i]:
                    # Convert to PIL Image
                    pil_image = numpy_to_pil(img_array)
                    
                    # Display image
                    st.image(
                        pil_image, 
                        caption=f"Sample {i+1}",
                        use_column_width=True
                    )
                    
                    # Add download button
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download",
                        data=img_buffer.getvalue(),
                        file_name=f"digit_{selected_digit}_sample_{i+1}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
    
    else:
        # Default display
        st.info("👆 Select a digit and click 'Generate Images' to create handwritten digit samples")
        
        # Show example
        st.subheader("Generate Images from Sidebar")
        st.markdown("Here's where the generated images will show up:")
        
        # Create placeholder images to show the layout
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                # Create a sample placeholder image
                placeholder_img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
                st.image(
                    placeholder_img, 
                    caption=f"Sample {i+1}",
                    use_column_width=True
                )


if __name__ == "__main__":
    main()
