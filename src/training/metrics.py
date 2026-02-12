import torch


def compute_batch_alignment(logits_per_image, logits_per_text):
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)
    img_to_text = (logits_per_image.argmax(dim=1) == targets).float().mean()
    text_to_img = (logits_per_text.argmax(dim=1) == targets).float().mean()
    batch_acc = (img_to_text + text_to_img) * 0.5
    return img_to_text, text_to_img, batch_acc
