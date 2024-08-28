import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from nbeats_pytorch.model import NBeatsNet
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Invoice Date'], index_col='Invoice Date')
    df = df.asfreq('W-MON')
    return df

def handle_outliers(series, method='iqr', factor=1.5):
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        series = np.where(series > upper_bound, upper_bound, series)
        series = np.where(series < lower_bound, lower_bound, series)
    return series

class TimeSeriesDataset(Dataset):
    def __init__(self, y, input_size, output_size):
        self.y = y
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.y) - self.input_size - self.output_size + 1

    def __getitem__(self, idx):
        x = self.y[idx:idx + self.input_size]
        y = self.y[idx + self.input_size:idx + self.input_size + self.output_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(train_loader, val_loader, input_size, output_size):
    model = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=output_size,
        backcast_length=input_size,
        hidden_layer_units=64,
        thetas_dim=(4, 4)
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping parameters
    early_stopping_patience = 20
    best_val_loss = float('inf')
    early_stopping_counter = 0
    epochs = 100
    # Initialize best_model_state with the current state of the model
    best_model_state = model.state_dict()

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            _, forecast = model(x_batch)
            loss = criterion(forecast, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation step
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                _, val_forecast = model(x_val)
                val_loss += criterion(val_forecast, y_val).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                model.load_state_dict(best_model_state)
                break

    return model

def make_forecast(model, y_train, input_size):
    model.eval()
    with torch.no_grad():
        _, forecast = model(torch.tensor(y_train[-input_size:], dtype=torch.float32).unsqueeze(0))
        return forecast.numpy().flatten()

def predict(file_path):
    set_seed(42)
    input_size = 52

    # Load and preprocess data
    df = load_data(file_path)
    df['Qty Invoiced'] = handle_outliers(df['Qty Invoiced'])

    # Print the preprocessed data
    print("Data used for training:")
    print(df)

    split_index = int(len(df) * 0.9)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    y_train = train['Qty Invoiced'].values
    y_test = test['Qty Invoiced'].values

    # Prepare the dataset and data loader
    output_size = len(y_test)
    full_dataset = TimeSeriesDataset(y_train, input_size, output_size)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model
    model = train_model(train_loader, val_loader, input_size, output_size)

    # Make the forecast
    forecast = make_forecast(model, y_train, input_size)
    test_mape = mean_absolute_percentage_error(y_test, forecast)

    return test, forecast, test_mape
