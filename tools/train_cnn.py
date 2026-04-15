import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ocr.cnn_recognizer import OcrDigitCNN

def main():
    base_dir = '../data/cnn_dataset' if os.path.exists('../data/cnn_dataset') else 'data/cnn_dataset'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    models_dir = '../models' if os.path.exists('../models') else 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_save_path = os.path.join(models_dir, 'cnn_digit_best.pth')
    
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"Error: {train_dir} 디렉토리가 없거나 비어 있습니다.")
        print("먼저 tools/augment_cnn_data.py 를 실행하여 학습 데이터를 준비해주세요.")
        return

    # Hyperparameters
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.001
    
    # 데이터셋 로더용 이미지 변환 룰
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # -1.0 ~ 1.0 정규화
    ])
    
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    print(f"Detected {num_classes} classes: {train_dataset.class_to_idx}")
    
    # Device Configuration (Mac M1/M2 MPS 지원 포함)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using compute device: {device}")
    
    # 모델 초기화
    model = OcrDigitCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    
    print("\nStarting Training Loop...")
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
              
        # 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            # 최종 추론 시 클래스 매핑(class_to_idx) 정보가 필요하므로 가중치와 함께 딕셔너리로 묶어 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx
            }, model_save_path)
            print(f" ---> Best Model Saved! (Acc: {best_acc:.2f}%)")

    print(f"\nTraining Finished! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.abspath(model_save_path)}")

if __name__ == '__main__':
    main()
