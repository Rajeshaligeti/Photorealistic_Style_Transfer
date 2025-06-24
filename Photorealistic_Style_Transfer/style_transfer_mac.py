import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import streamlit as st
import matplotlib.pyplot as plt
import time
import os
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

# Device setup optimized for M2
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# VGG Layer Mapping for human-readable block names
VGG_LAYER_MAP = {
    'block1_conv1': '0',
    'block1_conv2': '2',
    'block2_conv1': '5',
    'block2_conv2': '7',
    'block3_conv1': '10',
    'block3_conv2': '12',
    'block3_conv3': '14',
    'block3_conv4': '16',
    'block4_conv1': '19',
    'block4_conv2': '21',
    'block5_conv1': '28',
}

# Refined style and content layers optimized for photorealism
custom_layers = {
    VGG_LAYER_MAP['block1_conv1']: 'conv1_1',
    VGG_LAYER_MAP['block2_conv1']: 'conv2_1',
    VGG_LAYER_MAP['block3_conv1']: 'conv3_1',
    VGG_LAYER_MAP['block4_conv1']: 'conv4_1',
    VGG_LAYER_MAP['block5_conv1']: 'conv5_1',
    VGG_LAYER_MAP['block4_conv2']: 'conv4_2',  # content layer
}

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + \
            np.array((0.485, 0.456, 0.406))
    return np.clip(image, 0, 1)

def load_image(image_path, max_size=1024, shape=None):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image)
    return image.unsqueeze(0).to(device)

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def photorealistic_style_transfer(content_path, style_path,
                                 num_steps=500,  # Optimized for M2
                                 content_weight=0.3,  # Reduced to allow more style transfer
                                 style_weight=3e5,  # Increased for stronger style
                                 tv_weight=5e-8,  # Reduced for smoother results
                                 max_size=1024,  # Maximum size for highest quality
                                 show_every=100,  # Show results more frequently
                                 post_filter=True):
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        stop_button_container = st.empty()
        intermediate_results = st.empty()
        loss_metrics = st.empty()
        
        # Create stop button
        stop_button = stop_button_container.button("Stop Processing", key="stop_button_main")
        
        # Load pre-trained VGG19 model
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad_(False)

        # Load images with high-quality resizing
        content = load_image(content_path, max_size=max_size)
        style = load_image(style_path, shape=content.shape[-2:])

        # Initialize target image with content
        target = content.clone().requires_grad_(True)

        # Extract features
        content_features = get_features(content, vgg, custom_layers)
        style_features = get_features(style, vgg, custom_layers)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        # Optimizer and Scheduler optimized for M2
        optimizer = optim.Adam([target], lr=0.02)  # Increased learning rate
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        # Best loss tracking and history
        best_loss = float('inf')
        loss_history = []
        content_loss_history = []
        style_loss_history = []
        tv_loss_history = []
        best_result = None

        # Show initial state
        with metrics_container:
            st.write("### Initial State")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.imshow(im_convert(target))
            ax1.set_title('Initial Image')
            ax1.axis('off')
            ax2.plot(loss_history)
            ax2.set_title('Loss History')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            st.pyplot(fig)
            plt.close(fig)

        # Optimization loop
        start_time = time.time()
        for step in range(num_steps):
            if stop_button:
                status_text.info("Stopping process...")
                break
                
            # Update progress
            progress = (step + 1) / num_steps
            progress_bar.progress(progress)
            
            # Closure function
            def closure():
                nonlocal best_loss, best_result
                optimizer.zero_grad()

                # Extract features of the target image
                target_features = get_features(target, vgg, custom_layers)

                # Content loss (L2 distance between content features)
                content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

                # Style loss (Gram matrix-based loss for style features)
                style_loss = 0
                for layer in style_grams:
                    target_feature = target_features[layer]
                    target_gram = gram_matrix(target_feature)
                    _, d, h, w = target_feature.shape
                    style_gram = style_grams[layer]
                    layer_style_loss = torch.mean((target_gram - style_gram) ** 2) / (d * h * w)
                    style_loss += layer_style_loss / len(style_grams)

                # Total Variation loss (smooths the image)
                tv_loss = torch.sum(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])) + \
                          torch.sum(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]))

                # Total loss: weighted sum of content, style, and TV losses
                loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
                loss.backward()

                # Track loss history
                loss_history.append(loss.item())
                content_loss_history.append(content_loss.item())
                style_loss_history.append(style_loss.item())
                tv_loss_history.append(tv_loss.item())
                
                # Update best result
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_result = target.clone()
                    
                return loss

            # Update target image
            optimizer.step(closure)
            scheduler.step()

            # Display loss metrics in a more prominent way
            with loss_metrics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Loss", f"{loss_history[-1]:.2f}")
                with col2:
                    st.metric("Content Loss", f"{content_loss_history[-1]:.2f}")
                with col3:
                    st.metric("Style Loss", f"{style_loss_history[-1]:.2f}")
                with col4:
                    st.metric("TV Loss", f"{tv_loss_history[-1]:.2f}")
            
            status_text.info(f"Step {step + 1}/{num_steps} - Elapsed time: {time.time() - start_time:.1f}s")
            
            # Show intermediate results more frequently
            if (step + 1) % show_every == 0:
                with intermediate_results:
                    st.write(f"### Intermediate Result at Step {step + 1}")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.imshow(im_convert(target))
                    ax1.set_title(f'Step {step + 1}')
                    ax1.axis('off')
                    
                    # Plot all losses
                    ax2.plot(loss_history, label='Total Loss', linewidth=2)
                    ax2.plot(content_loss_history, label='Content Loss', linewidth=2)
                    ax2.plot(style_loss_history, label='Style Loss', linewidth=2)
                    ax2.plot(tv_loss_history, label='TV Loss', linewidth=2)
                    ax2.set_title('Loss History', fontsize=12)
                    ax2.set_xlabel('Step', fontsize=10)
                    ax2.set_ylabel('Loss', fontsize=10)
                    ax2.legend(fontsize=10)
                    ax2.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Check stop button state
            stop_button = stop_button_container.button("Stop Processing", key=f"stop_button_{step}")

        # Show final state
        with metrics_container:
            st.write("### Final Result")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.imshow(im_convert(target))
            ax1.set_title('Final Image')
            ax1.axis('off')
            
            # Plot all losses
            ax2.plot(loss_history, label='Total Loss', linewidth=2)
            ax2.plot(content_loss_history, label='Content Loss', linewidth=2)
            ax2.plot(style_loss_history, label='Style Loss', linewidth=2)
            ax2.plot(tv_loss_history, label='TV Loss', linewidth=2)
            ax2.set_title('Loss History', fontsize=12)
            ax2.set_xlabel('Step', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.legend(fontsize=10)
            ax2.grid(True)
            st.pyplot(fig)
            plt.close(fig)

        # Clean up UI elements
        progress_bar.empty()
        status_text.empty()
        stop_button_container.empty()
        metrics_container.empty()
        intermediate_results.empty()
        loss_metrics.empty()

        # Return the best result if available, otherwise the final result
        return best_result if best_result is not None else target

    except Exception as e:
        # Clean up UI elements in case of error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        if 'stop_button_container' in locals():
            stop_button_container.empty()
        if 'metrics_container' in locals():
            metrics_container.empty()
        if 'intermediate_results' in locals():
            intermediate_results.empty()
        if 'loss_metrics' in locals():
            loss_metrics.empty()
        raise e

def save_output(output_tensor, output_path):
    """Save the output tensor as an image with high quality."""
    output_image = im_convert(output_tensor)
    output_image = (output_image * 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    # Save with maximum quality
    output_image.save(output_path, quality=100, subsampling=0) 