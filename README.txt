# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
# Linux / Mac
source venv/bin/activate

# Windows
# venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the program
python main.py


python predict.py data.csv --model gas_model_v2.pkl --window 50 --step 25 --smooth 7 --smooth_pred_win 3