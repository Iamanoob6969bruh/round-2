import modal

app = modal.App("test-gpu")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "easyocr")
)

@app.function(image=image, gpu="T4")
def run_test():
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
