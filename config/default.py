import os 

# Change your input size here
input_image_size = 256

# Change your batch size here
batch_size = 4

# Change your epoch here
epoch = 5

# Change your train image root path here
train_img_path = "/datasets/coco/train2014/"

# Change your train annot json path here
train_annot_path = "/datasets/coco/annotations/captions_train2014.json"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your diffusion prior model save path here (end with ".pth")
diff_save_path = "/storage/dalle-2/diffusion/diff_prior.pth"

# Change your diffusion prior model save path here (end with ".pth")
decoder_save_path = "/storage/dalle-2/decoder/decoder.pth"

# Change the model weight save path here (end with ".pth")
dalle2_save_path = "/storage/dalle-2/dalle2.pth"

# Change the test result image save path (should be a directory or folder)
test_img_save_path = "./result"

if not os.path.exists(test_img_save_path):
    os.makedirs(test_img_save_path)