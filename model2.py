import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import json
import time
import os

try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    pass

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis and Repeat Purchase Predictor")
        self.root.state("zoomed") 
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.file_path = tk.StringVar()
        self.tagalog_sentiment = {
            'lupet': 3, 'galing': 3.2, 'ayos': 2.7, 'maganda': 3.7,
            'ganda': 3.4, 'ayos': 3.2, 'maayos': 3.5, 'angas': 2.8,
            'panalo': 3.5, 'mabilis': 3.2, 'bilis': 2.8, 'pangit': -3.4,
            'di ko gusto': -3.5, 'di gusto': -3.3, 'panget': -3.4,
            'ayaw': -3.1, 'malungkot': -2.8, 'lungkot': -2.8,
            'tapon': -3.4, 'tinapon': -3.2
        }
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        self.sia.lexicon.update(self.tagalog_sentiment)
        
        # Results storage
        self.model_results = {}
        self.regression = None
        self.scaler = None
        self.amazon_reviews = None
        self.shopee_reviews = None
        self.lazada_reviews = None
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Data Selection", padding="10")
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Amazon reviews file
        ttk.Label(file_frame, text="Amazon Reviews:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.amazon_entry = ttk.Entry(file_frame, width=50)
        self.amazon_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(file_frame, text="Browse...", command=lambda: self.browse_file('amazon')).grid(row=0, column=2, padx=5, pady=5)
        
        # Shopee reviews file
        ttk.Label(file_frame, text="Shopee Reviews:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.shopee_entry = ttk.Entry(file_frame, width=50)
        self.shopee_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(file_frame, text="Browse...", command=lambda: self.browse_file('shopee')).grid(row=1, column=2, padx=5, pady=5)
        
        # Lazada reviews file
        ttk.Label(file_frame, text="Lazada Reviews:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.lazada_entry = ttk.Entry(file_frame, width=50)
        self.lazada_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(file_frame, text="Browse...", command=lambda: self.browse_file('lazada')).grid(row=2, column=2, padx=5, pady=5)
        
        # Analysis options frame
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="10")
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Monte Carlo simulations
        ttk.Label(options_frame, text="Monte Carlo Simulations:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mc_sims = ttk.Spinbox(options_frame, from_=100, to=50000, increment=100, width=10)
        self.mc_sims.set(1000)
        self.mc_sims.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Test size
        ttk.Label(options_frame, text="Test Size (%):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.test_size_percent = ttk.Spinbox(options_frame, from_=5, to=50, increment=5, width=10)
        self.test_size_percent.set(20)
        self.test_size_percent.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress = ttk.Progressbar(controls_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.analyze_btn = ttk.Button(controls_frame, text="Analyze Data", command=self.run_analysis)
        self.analyze_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Notebook for results
        self.results_notebook = ttk.Notebook(main_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results tab
        self.results_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.results_tab, text="Model Results")
        
        # Plots tabs
        self.lr_plot_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.lr_plot_tab, text="Logistic Regression")
        
        self.mc_plot_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.mc_plot_tab, text="Monte Carlo Simulation")
        
        self.coef_plot_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.coef_plot_tab, text="Coefficient Distribution")
        
        # Results text area
        self.results_text = tk.Text(self.results_tab, height=20, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.statusbar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_file(self, file_type):
        filetypes = []
        if file_type == 'amazon' or file_type == 'shopee':
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        elif file_type == 'lazada':
            filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        
        filename = filedialog.askopenfilename(
            title=f"Select {file_type.capitalize()} Reviews File",
            filetypes=filetypes
        )
        
        if filename:
            if file_type == 'amazon':
                self.amazon_entry.delete(0, tk.END)
                self.amazon_entry.insert(0, filename)
            elif file_type == 'shopee':
                self.shopee_entry.delete(0, tk.END)
                self.shopee_entry.insert(0, filename)
            elif file_type == 'lazada':
                self.lazada_entry.delete(0, tk.END)
                self.lazada_entry.insert(0, filename)
    
    def run_analysis(self):
        # Check if at least one file is selected
        if not self.amazon_entry.get() and not self.shopee_entry.get() and not self.lazada_entry.get():
            messagebox.showerror("Error", "Please select at least one data file")
            return
        
        self.status_var.set("Loading data...")
        self.progress['value'] = 10
        self.root.update_idletasks()
        
        # Load data
        try:
            # Load Amazon data if available
            if self.amazon_entry.get():
                self.amazon_reviews = pd.read_csv(self.amazon_entry.get())
            else:
                self.amazon_reviews = pd.DataFrame({'reviewText': [], 'overall': []})
            
            # Load Shopee data if available
            if self.shopee_entry.get():
                self.shopee_reviews = pd.read_csv(self.shopee_entry.get())
            else:
                self.shopee_reviews = pd.DataFrame({'review': [], 'rating': []})
            
            # Load Lazada data if available
            if self.lazada_entry.get():
                self.lazada_reviews = pd.read_json(self.lazada_entry.get())
            else:
                self.lazada_reviews = pd.DataFrame({'review': [], 'rating': []})
            
            # Check if required columns exist
            if 'reviewText' not in self.amazon_reviews.columns or 'overall' not in self.amazon_reviews.columns:
                if not self.amazon_reviews.empty:
                    messagebox.showwarning("Warning", "Amazon data missing required columns (reviewText, overall)")
            
            if 'review' not in self.shopee_reviews.columns or 'rating' not in self.shopee_reviews.columns:
                if not self.shopee_reviews.empty:
                    messagebox.showwarning("Warning", "Shopee data missing required columns (review, rating)")
            
            if 'review' not in self.lazada_reviews.columns or 'rating' not in self.lazada_reviews.columns:
                if not self.lazada_reviews.empty:
                    messagebox.showwarning("Warning", "Lazada data missing required columns (review, rating)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
            return
        
        # Continue with analysis
        self.status_var.set("Processing sentiment analysis...")
        self.progress['value'] = 20
        self.root.update_idletasks()
        
        # Run the analysis in a separate thread to prevent UI freezing
        self.root.after(100, self.perform_analysis)
    
    def perform_analysis(self):
        try:
            # Combine all reviews for sentiment analysis
            amazon_text = self.amazon_reviews['reviewText'] if 'reviewText' in self.amazon_reviews.columns else pd.Series([])
            shopee_text = self.shopee_reviews['review'].head(5000) if 'review' in self.shopee_reviews.columns else pd.Series([])
            lazada_text = self.lazada_reviews['review'] if 'review' in self.lazada_reviews.columns else pd.Series([])
            
            overall_reviews = pd.concat([amazon_text, shopee_text, lazada_text], ignore_index=True)
            sentiment_score = overall_reviews.astype(str).apply(lambda text: self.sia.polarity_scores(text)['compound'])
            
            self.status_var.set("Building dataset...")
            self.progress['value'] = 30
            self.root.update_idletasks()
            
            amazon_ratings = self.amazon_reviews['overall'] if 'overall' in self.amazon_reviews.columns else pd.Series([])
            shopee_ratings = self.shopee_reviews['rating'].head(5000) if 'rating' in self.shopee_reviews.columns else pd.Series([])
            lazada_ratings = self.lazada_reviews['rating'] if 'rating' in self.lazada_reviews.columns else pd.Series([])
            
            sentiments = pd.DataFrame({
                'sentiment_score': sentiment_score,
                'ratings': pd.concat([amazon_ratings, shopee_ratings, lazada_ratings], ignore_index=True)
            })
            
            # Set repeat_buy based on sentiment and ratings
            sentiments['repeat_buy'] = 0
            
            sentiments.loc[
                (sentiments['sentiment_score'] >= 0.05) & (sentiments['ratings'] >= 4),
                'repeat_buy'
            ] = 1  # likely to buy again
            
            sentiments.loc[
                (sentiments['sentiment_score'] <= -0.05) & (sentiments['ratings'] <= 2),
                'repeat_buy'
            ] = 0  # less likely to buy again
            
            sentiments.loc[
                (sentiments['sentiment_score'].between(-1.75, 1.75)) & (sentiments['ratings'] == 3),
                'repeat_buy'
            ] = 2  # 50/50
            
            # Remove rows with NaN values
            sentiments = sentiments[sentiments['repeat_buy'].notna()]
            
            if sentiments.empty:
                messagebox.showerror("Error", "No valid data after processing. Please check your input files.")
                self.status_var.set("Error: No valid data")
                return
                
            self.status_var.set("Training model...")
            self.progress['value'] = 40
            self.root.update_idletasks()
            
            # Prepare data for model
            x = sentiments[['sentiment_score', 'ratings']]
            y = sentiments['repeat_buy']
            
            # Get test size as a percentage and convert to decimal
            try:
                test_size_percent = float(self.test_size_percent.get())
                test_size = test_size_percent / 100.0  # Convert percentage to decimal
            except:
                test_size = 0.2  # Default to 20% if there's an error
                
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=42
            )
            
            # Scale data
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)
            
            # Train logistic regression
            self.regression = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs'
            )
            
            self.regression.fit(x_train_scaled, y_train)
            
            # Evaluate model
            y_result = self.regression.predict(x_test_scaled)
            accuracy = accuracy_score(y_test, y_result)
            mae = mean_absolute_error(y_test, y_result)
            report = classification_report(y_test, y_result, zero_division=0, output_dict=True)
            
            # Store results
            self.model_results = {
                'accuracy': accuracy,
                'mae': mae,
                'report': report,
                'test_data': x_test,
                'test_labels': y_test,
                'predictions': y_result,
                'coefficients': self.regression.coef_[0],
                'feature_names': x.columns.tolist(),
                'original_intercept': self.regression.intercept_[0]
            }
            
            # Display results
            self.update_results_text()
            
            # Monte Carlo simulation
            self.status_var.set("Running Monte Carlo simulation...")
            self.progress['value'] = 50
            self.root.update_idletasks()
            
            try:
                simulations = int(self.mc_sims.get()) 
            except:
                simulations = 1000
                
            # Calculate test points based on the number of simulations
            n_test_points = int(simulations * (test_size))  # Use the same test size percentage
            
            self.run_monte_carlo(simulations, x_test, y_test, n_test_points)
            
            # Create plots
            self.status_var.set("Creating visualizations...")
            self.progress['value'] = 80
            self.root.update_idletasks()
            
            self.create_lr_plot(x_test, y_test)
            self.create_mc_plot()
            self.create_coef_plot()
            
            self.status_var.set("Analysis completed successfully")
            self.progress['value'] = 100
            self.root.update_idletasks()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")
            import traceback
            traceback.print_exc()
    
    def update_results_text(self):
        """Update the results text area with model performance metrics"""
        self.results_text.delete(1.0, tk.END)
        
        results = self.model_results
        report = results['report']
        
        # Format classification report
        self.results_text.insert(tk.END, "Classification Report:\n\n")
        header = f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
        self.results_text.insert(tk.END, header)
        
        for class_id, metrics in report.items():
            if class_id in ['0', '1', '2']:
                line = f"{class_id:>10} {metrics['precision']:>10.2f} {metrics['recall']:>10.2f} {metrics['f1-score']:>10.2f} {metrics['support']:>10}\n"
                self.results_text.insert(tk.END, line)
        
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"{'accuracy':>10}\n")
        self.results_text.insert(tk.END, f"{'macro avg':>10} {report['macro avg']['precision']:>10.2f} {report['macro avg']['recall']:>10.2f} {report['macro avg']['f1-score']:>10.2f} {report['macro avg']['support']:>10}\n")
        self.results_text.insert(tk.END, f"{'weighted avg':>10} {report['weighted avg']['precision']:>10.2f} {report['weighted avg']['recall']:>10.2f} {report['weighted avg']['f1-score']:>10.2f} {report['weighted avg']['support']:>10}\n")
        
        self.results_text.insert(tk.END, f"\nMean Absolute Error: {results['mae']:.4f}\n")
        self.results_text.insert(tk.END, f"\nCoefficients:\n")
        
        for feature, coef in zip(results['feature_names'], results['coefficients']):
            self.results_text.insert(tk.END, f"{feature}: {coef:.4f}\n")
        
        self.results_text.insert(tk.END, f"\nAccuracy of the model: {results['accuracy']:.2%}\n")
    
    def run_monte_carlo(self, simulations, x_test, y_test, n_test_points=None):
        """Run Monte Carlo simulation"""
        start = time.time()
        
        mc_results = []
        
        original_coefficients = self.model_results['coefficients']
        original_intercept = self.model_results['original_intercept']
        
        coefficient_std = 0.1
        intercept_std = 0.1
        
        # If n_test_points is not provided, calculate it as 20% of simulations
        if n_test_points is None:
            try:
                test_size_percent = float(self.test_size_percent.get())
                n_test_points = int(simulations * (test_size_percent / 100.0))
            except:
                n_test_points = int(simulations * 0.2)  # Default to 20%
        
        # Ensure minimum number of test points
        n_test_points = max(n_test_points, 50)
        
        test_sentiments = np.random.uniform(-1, 1, n_test_points)
        test_ratings = np.random.randint(1, 6, n_test_points)
        test_data = pd.DataFrame({
            'sentiment_score': test_sentiments,
            'ratings': test_ratings
        })
        test_data_scaled = self.scaler.transform(test_data)
        
        all_predictions = np.zeros((simulations, n_test_points))
        
        progress_interval = max(1, simulations // 20)  # Update progress every 5%
        
        for sim in range(simulations):
            sampled_coefficients = np.random.normal(
                original_coefficients, 
                coefficient_std, 
                size=original_coefficients.shape
            )
            sampled_intercept = np.random.normal(
                original_intercept, 
                intercept_std
            )
            
            mc_model = LogisticRegression()
            mc_model.classes_ = self.regression.classes_
            mc_model.coef_ = np.array([sampled_coefficients])
            mc_model.intercept_ = np.array([sampled_intercept])
            
            predictions = mc_model.predict(test_data_scaled)
            all_predictions[sim] = predictions
            
            if sim < 10:
                for i in range(min(3, n_test_points)):
                    mc_results.append({
                        'simulation': sim,
                        'sentiment_score': test_data.iloc[i]['sentiment_score'],
                        'rating': test_data.iloc[i]['ratings'],
                        'prediction': predictions[i]
                    })
            
            # Update progress
            if sim % progress_interval == 0:
                progress_value = 50 + (sim / simulations) * 30
                self.progress['value'] = progress_value
                self.root.update_idletasks()
        
        end = time.time()
        
        prob_class_0 = np.mean(all_predictions == 0, axis=0)
        prob_class_1 = np.mean(all_predictions == 1, axis=0)
        prob_class_2 = np.mean(all_predictions == 2, axis=0)
        
        test_data['prob_class_0'] = prob_class_0
        test_data['prob_class_1'] = prob_class_1
        test_data['prob_class_2'] = prob_class_2
        
        test_data['most_likely_class'] = test_data[['prob_class_0', 'prob_class_1', 'prob_class_2']].idxmax(axis=1)
        test_data['most_likely_class'] = test_data['most_likely_class'].map({
            'prob_class_0': 0,
            'prob_class_1': 1,
            'prob_class_2': 2
        })
        
        test_data['uncertainty'] = -(
            test_data['prob_class_0'] * np.log2(test_data['prob_class_0'] + 1e-10) +
            test_data['prob_class_1'] * np.log2(test_data['prob_class_1'] + 1e-10) +
            test_data['prob_class_2'] * np.log2(test_data['prob_class_2'] + 1e-10)
        )
        
        # Store Monte Carlo results
        self.model_results['mc_results'] = mc_results
        self.model_results['mc_test_data'] = test_data
        self.model_results['mc_execution_time'] = end - start
        self.model_results['n_test_points'] = n_test_points
        self.model_results['simulations'] = simulations
        
        # Update results text with Monte Carlo info
        # self.results_text.insert(tk.END, f"\n--- Monte Carlo Simulation ---\n")
        # self.results_text.insert(tk.END, f"Simulations: {simulations}\n")
        # self.results_text.insert(tk.END, f"Test Points: {n_test_points} ({int(100 * n_test_points / simulations)}% of simulations)\n")
        # self.results_text.insert(tk.END, "\nSample Monte Carlo Results:\n")
        
        for i in range(min(5, len(mc_results))):
            sim = mc_results[i]['simulation']
            sentiment = mc_results[i]['sentiment_score']
            rating = mc_results[i]['rating']
            prediction = mc_results[i]['prediction']
            label = {0: "Less likely to buy again", 1: "Likely to buy again", 2: "Unsure Buyer"}[prediction]
            # self.results_text.insert(tk.END, f"Simulation {sim}, Sentiment: {sentiment:.2f}, Rating: {rating}, Prediction: {label}\n")

        # self.results_text.insert(tk.END, "\nMonte Carlo Simulation Summary:\n")
        # self.results_text.insert(tk.END, f"Less likely to buy again (Class 0): {np.sum(test_data['most_likely_class'] == 0)} points\n")
        # self.results_text.insert(tk.END, f"Likely to buy again (Class 1): {np.sum(test_data['most_likely_class'] == 1)} points\n")
        # self.results_text.insert(tk.END, f"Unsure Buyer (Class 2): {np.sum(test_data['most_likely_class'] == 2)} points\n")
        # self.results_text.insert(tk.END, f"\nExecution Speed ({simulations} simulations): {end-start:.2f} seconds\n")
    
    def create_lr_plot(self, x_test, y_test):
        """Create logistic regression decision boundary plot"""
        # Clear previous plot
        for widget in self.lr_plot_tab.winfo_children():
            widget.destroy()
            
        # Create figure
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create decision boundary plot
        h = 0.01
        x_min, x_max = x_test['sentiment_score'].min() - 0.1, x_test['sentiment_score'].max() + 0.1
        y_min, y_max = x_test['ratings'].min() - 0.5, x_test['ratings'].max() + 0.5
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = self.scaler.transform(grid)
        
        Z = self.regression.predict(grid_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        scatter = ax.scatter(x_test['sentiment_score'], x_test['ratings'], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Ratings')
        ax.set_title('Logistic Regression Decision Boundary')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Repeat Buy (0 = No, 1 = Yes, 2 = Unsure)')
        ax.grid(True)
        
        # Add to tab
        canvas = FigureCanvasTkAgg(fig, master=self.lr_plot_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_mc_plot(self):
        """Create Monte Carlo simulation plot"""
        # Clear previous plot
        for widget in self.mc_plot_tab.winfo_children():
            widget.destroy()
            
        # Get data
        if 'mc_test_data' not in self.model_results:
            return
            
        test_data = self.model_results['mc_test_data']
        n_test_points = self.model_results.get('n_test_points', 'N/A')
        simulations = self.model_results.get('simulations', 'N/A')
        
        # Create figure
        fig = plt.Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot
        scatter = ax.scatter(
            test_data['sentiment_score'],
            test_data['ratings'],
            c=test_data['most_likely_class'],
            s=test_data['uncertainty'] * 100,
            alpha=0.6,
            cmap=plt.cm.Paired,
            edgecolor='black'
        )
        
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Rating')
        ax.set_title(f'Monte Carlo Simulation Results (Simulations: {simulations}, Test Points: {n_test_points})')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Class')
        ax.grid(True)
        
        # Add legend
        sizes = [0.2, 0.5, 1.0]
        labels = ['Low uncertainty', 'Medium uncertainty', 'High uncertainty']
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    label=labels[i], 
                                    markerfacecolor='gray', 
                                    markersize=np.sqrt(sizes[i] * 100)) 
                        for i in range(len(sizes))]
        ax.legend(handles=legend_elements, title='Prediction Uncertainty')
        
        # Add to tab
        canvas = FigureCanvasTkAgg(fig, master=self.mc_plot_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_coef_plot(self):
        """Create coefficient distribution plot"""
        # Clear previous plot
        for widget in self.coef_plot_tab.winfo_children():
            widget.destroy()
            
        # Get data
        original_coefficients = self.model_results['coefficients']
        feature_names = self.model_results['feature_names']

        coefficient_std = 0.1
        coefficient_samples = np.random.normal(
            original_coefficients.reshape(-1, 1), 
            coefficient_std, 
            size=(len(original_coefficients), 1000)
        )
        
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        for i, feature in enumerate(feature_names):
            ax = fig.add_subplot(len(feature_names), 1, i+1)
            sns.histplot(coefficient_samples[i], kde=True, ax=ax)
            ax.axvline(original_coefficients[i], color='r', linestyle='--')
            ax.set_title(f'Distribution of {feature} Coefficient')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Frequency')
        
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.coef_plot_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()
                        