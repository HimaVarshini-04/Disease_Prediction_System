# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_python')

def run_python():
    import pandas as pd
    import tkinter as tk
    from tkinter import LEFT
    import numpy as np
    import sqlite3
    from tkinter import messagebox
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    # Load the training and testing datasets
    df = pd.read_csv('training.csv')
    df1 = pd.read_csv('testing.csv')

    # Extract symptom and disease columns
    cols = df.columns[:-1]
    symptoms = cols.values.tolist()
    diseases = df['prognosis'].unique()

    # Initialize symptom encoding
    l1 = [0] * len(symptoms)

    # Train models (in the background)
    model_dt = tree.DecisionTreeClassifier().fit(df[symptoms], df['prognosis'])
    model_rf = RandomForestClassifier().fit(df[symptoms], df['prognosis'])
    model_knn = KNeighborsClassifier(n_neighbors=5).fit(df[symptoms], df['prognosis'])
    model_nb = GaussianNB().fit(df[symptoms], df['prognosis'])

    # Tkinter GUI setup
    root = tk.Tk()
    root.title("Disease Prediction")
    root.geometry("600x600")
    root.configure(bg="#4682b4")

    # Define Variables
    Symptom1 = tk.StringVar(value="Select Here")
    Symptom2 = tk.StringVar(value="Select Here")
    Symptom3 = tk.StringVar(value="Select Here")
    Symptom4 = tk.StringVar(value="Select Here")
    Symptom5 = tk.StringVar(value="Select Here")
    pred1 = tk.StringVar()
    precaution_var = tk.StringVar()
    medication_var = tk.StringVar()

    # Instruction Label
    tk.Label(root, text="Please enter the symptoms you have", bg="#4682b4", fg="white", font=("Helvetica", 14)).grid(row=0,
                                                                                                                  column=0,
                                                                                                                  padx=10,
                                                                                                                  pady=10,
                                                                                                                  columnspan=2)

    # Create Symptom Selection Options
    tk.Label(root, text="Symptom 1", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=1, column=0, padx=10,
                                                                                         pady=10)
    tk.OptionMenu(root, Symptom1, *symptoms).grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Symptom 2", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=2, column=0, padx=10,
                                                                                         pady=10)
    tk.OptionMenu(root, Symptom2, *symptoms).grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Symptom 3", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=3, column=0, padx=10,
                                                                                         pady=10)
    tk.OptionMenu(root, Symptom3, *symptoms).grid(row=3, column=1, padx=10, pady=10)

    tk.Label(root, text="Symptom 4", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=4, column=0, padx=10,
                                                                                         pady=10)
    tk.OptionMenu(root, Symptom4, *symptoms).grid(row=4, column=1, padx=10, pady=10)

    tk.Label(root, text="Symptom 5", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=5, column=0, padx=10,
                                                                                         pady=10)
    tk.OptionMenu(root, Symptom5, *symptoms).grid(row=5, column=1, padx=10, pady=10)

    # Prediction function
    def predict():
        symptoms_input = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
        input_vector = [1 if symptom in symptoms_input else 0 for symptom in symptoms]

        # Using one of the trained models (e.g., RandomForest)
        prediction = model_rf.predict([input_vector])
        pred1.set(prediction[0])

    # Precautions and Medications dictionary
    precaution_dict = {
        "Fungal infection": ["Keep affected area clean and dry", "Use antifungal creams",
                             "Avoid sharing personal items", "Wear breathable clothing"],
        "Allergy": ["Avoid allergens", "Take antihistamines", "Use allergy-proof bedding",
                    "Keep windows closed during high pollen season"],
        "Peptic ulcer disease": ["Avoid spicy foods", "Reduce stress", "Avoid NSAIDs", "Eat smaller, frequent meals"],
        "AIDS": ["Take antiretroviral medications", "Avoid exposure to infections", "Maintain a healthy diet",
                 "Regular medical check-ups"],
        "Diabetes": ["Monitor blood sugar levels", "Maintain a balanced diet", "Exercise regularly",
                     "Take prescribed medications"],
        "Gastroenteritis": ["Stay hydrated", "Avoid dairy products", "Wash hands frequently", "Eat small, bland meals"],
        "Bronchial Asthma": ["Avoid triggers", "Use inhalers as prescribed", "Monitor lung function",
                             "Practice breathing exercises"],
        "Hypertension": ["Limit salt intake", "Exercise regularly", "Avoid excessive alcohol", "Manage stress"],
        "Migraine": ["Avoid triggers", "Take prescribed medications", "Rest in a dark, quiet room", "Stay hydrated"],
        "Cervical spondylosis": ["Exercise regularly", "Maintain good posture", "Use a supportive pillow",
                                 "Avoid heavy lifting"],
        "Paralysis (brain hemorrhage)": ["Immediate medical attention", "Physical therapy", "Occupational therapy",
                                         "Supportive care"],
        "Jaundice": ["Avoid alcohol", "Eat a healthy diet", "Get vaccinated for hepatitis",
                     "Avoid contaminated food and water"],
        "Malaria": ["Use mosquito nets", "Take antimalarial medication", "Wear long sleeves", "Use insect repellent"],
        "Chicken pox": ["Avoid scratching", "Use calamine lotion", "Keep fingernails trimmed", "Stay hydrated"],
        "Dengue": ["Use mosquito repellent", "Wear protective clothing", "Stay in well-screened areas",
                   "Avoid standing water"],
        "Typhoid": ["Avoid street food", "Drink bottled or boiled water", "Get vaccinated", "Practice good hygiene"],
        "Hepatitis A": ["Get vaccinated", "Practice good hygiene", "Avoid raw or undercooked food",
                        "Avoid contaminated water"],
        "Hepatitis B": ["Get vaccinated", "Practice safe sex", "Avoid sharing needles", "Avoid alcohol"],
        "Hepatitis C": ["Avoid sharing needles", "Practice safe sex", "Get regular liver function tests",
                        "Avoid alcohol"],
        "Hepatitis D": ["Get vaccinated against hepatitis B", "Avoid sharing needles", "Practice safe sex",
                        "Avoid alcohol"],
        "Hepatitis E": ["Avoid drinking contaminated water", "Practice good hygiene", "Avoid raw meat",
                        "Get vaccinated if available"],
        "Alcoholic hepatitis": ["Avoid alcohol", "Maintain a healthy diet", "Take prescribed medications",
                                "Regular liver function tests"],
        "Tuberculosis": ["Take all prescribed medications", "Avoid close contact with others",
                         "Cover mouth when coughing", "Regular medical check-ups"],
        "Common Cold": ["Rest and hydrate", "Use decongestants", "Gargle with salt water",
                        "Avoid close contact with others"],
        "Pneumonia": ["Take antibiotics as prescribed", "Rest and hydrate", "Use a humidifier", "Get vaccinated"],
        "Dimorphic hemorrhoids (piles)": ["Eat a high-fiber diet", "Stay hydrated",
                                          "Avoid straining during bowel movements", "Use over-the-counter creams"],
        "Heart attack": ["Take prescribed medications", "Adopt a heart-healthy diet", "Exercise regularly",
                         "Manage stress"],
        "Varicose veins": ["Exercise regularly", "Elevate your legs", "Avoid standing for long periods",
                           "Wear compression stockings"],
        "Hypothyroidism": ["Take thyroid hormone replacement", "Eat a balanced diet", "Regular thyroid function tests",
                           "Avoid excessive soy and iodine"],
        "Hyperthyroidism": ["Take antithyroid medications", "Avoid excessive iodine intake",
                            "Regular thyroid function tests", "Consider radioactive iodine therapy"],
        "Hypoglycemia": ["Eat small, frequent meals", "Carry glucose tablets", "Avoid excessive alcohol",
                         "Monitor blood sugar levels"],
        "Osteoarthritis": ["Exercise regularly", "Maintain a healthy weight", "Use hot or cold therapy",
                           "Take pain relievers as needed"],
        "Arthritis": ["Stay physically active", "Maintain a healthy weight", "Use assistive devices",
                      "Take anti-inflammatory medications"],
        "(vertigo) Paroymsal Positional Vertigo": ["Perform Epley maneuver", "Avoid sudden head movements",
                                                   "Stay hydrated", "Use vestibular rehabilitation exercises"],
        "Acne": ["Cleanse skin gently", "Avoid picking or squeezing pimples", "Use non-comedogenic products",
                 "Take prescribed medications"],
        "Urinary tract infection": ["Drink plenty of water", "Avoid holding urine", "Wipe from front to back",
                                    "Avoid irritating feminine products"],
        "Psoriasis": ["Moisturize regularly", "Avoid triggers like stress", "Use medicated creams",
                      "Get sunlight exposure"],
        "Impetigo": ["Keep affected area clean", "Avoid close contact with others", "Use prescribed antibiotics",
                     "Wash hands frequently"]
    }

    medication_dict = {
        "Fungal infection": ["Clotrimazole", "Miconazole", "Fluconazole", "Terbinafine"],
        "Allergy": ["Loratadine", "Cetirizine", "Fexofenadine", "Diphenhydramine"],
        "Peptic ulcer disease": ["Omeprazole", "Pantoprazole", "Ranitidine", "Antacids"],
        "AIDS": ["Antiretroviral therapy (ART)", "Efavirenz", "Lamivudine", "Zidovudine"],
        "Diabetes": ["Metformin", "Insulin", "Glipizide", "Pioglitazone"],
        "Gastroenteritis": ["Oral rehydration solution", "Loperamide", "Ondansetron", "Probiotics"],
        "Bronchial Asthma": ["Salbutamol inhaler", "Budesonide", "Montelukast", "Theophylline"],
        "Hypertension": ["Amlodipine", "Lisinopril", "Losartan", "Hydrochlorothiazide"],
        "Migraine": ["Sumatriptan", "Rizatriptan", "Ergotamine", "Propranolol"],
        "Cervical spondylosis": ["NSAIDs", "Muscle relaxants", "Physical therapy", "Gabapentin"],
        "Paralysis (brain hemorrhage)": ["Mannitol", "Antihypertensives", "Anticoagulants", "Physical therapy"],
        "Jaundice": ["Cholestyramine", "Ursodeoxycholic acid", "Vitamin K", "Lactulose"],
        "Malaria": ["Chloroquine", "Artemisinin", "Quinine", "Atovaquone"],
        "Chicken pox": ["Acyclovir", "Paracetamol", "Calamine lotion", "Antihistamines"],
        "Dengue": ["Paracetamol", "Oral rehydration solution", "Caripill", "Avoid NSAIDs"],
        "Typhoid": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone", "Chloramphenicol"],
        "Hepatitis A": ["No specific medication", "Supportive care", "Rest", "Hydration"],
        "Hepatitis B": ["Entecavir", "Tenofovir", "Lamivudine", "Peginterferon"],
        "Hepatitis C": ["Sofosbuvir", "Ledipasvir", "Ribavirin", "Daclatasvir"],
        "Hepatitis D": ["Peginterferon alfa", "No specific medication", "Supportive care", "Hepatitis B vaccine"],
        "Hepatitis E": ["No specific medication", "Supportive care", "Rest", "Hydration"],
        "Alcoholic hepatitis": ["Corticosteroids", "Pentoxifylline", "Abstinence from alcohol", "Nutritional support"],
        "Tuberculosis": ["Isoniazid", "Rifampin", "Ethambutol", "Pyrazinamide"],
        "Common Cold": ["Paracetamol", "Pseudoephedrine", "Dextromethorphan", "Saline nasal spray"],
        "Pneumonia": ["Amoxicillin", "Clarithromycin", "Ceftriaxone", "Azithromycin"],
        "Dimorphic hemorrhoids (piles)": ["Lidocaine", "Hydrocortisone", "Witch hazel", "Diosmin"],
        "Heart attack": ["Aspirin", "Clopidogrel", "Nitroglycerin", "Atorvastatin"],
        "Varicose veins": ["Compression stockings", "Diosmin", "Horse chestnut", "Sclerotherapy"],
        "Hypothyroidism": ["Levothyroxine", "Liothyronine", "Desiccated thyroid", "Regular monitoring"],
        "Hyperthyroidism": ["Methimazole", "Propylthiouracil", "Beta-blockers", "Radioactive iodine"],
        "Hypoglycemia": ["Glucose tablets", "Dextrose", "Glucagon injection", "Regular monitoring"],
        "Osteoarthritis": ["Acetaminophen", "Ibuprofen", "Glucosamine", "Topical NSAIDs"],
        "Arthritis": ["Methotrexate", "Sulfasalazine", "Hydroxychloroquine", "Leflunomide"],
        "(vertigo) Paroymsal Positional Vertigo": ["Meclizine", "Diazepam", "Epley maneuver",
                                                   "Vestibular rehabilitation"],
        "Acne": ["Benzoyl peroxide", "Salicylic acid", "Topical retinoids", "Oral antibiotics"],
        "Urinary tract infection": ["Trimethoprim-sulfamethoxazole", "Ciprofloxacin", "Nitrofurantoin", "Amoxicillin"],
        "Psoriasis": ["Topical corticosteroids", "Methotrexate", "Phototherapy", "Biologics"],
        "Impetigo": ["Mupirocin", "Retapamulin", "Cephalexin", "Amoxicillin/clavulanate"]
    }

    # Function to fetch precautions
    def get_precautions(disease):
        return precaution_dict.get(disease, ["No precautions available for this disease."])

    # Function to fetch medications
    def get_medications(disease):
        return medication_dict.get(disease, ["No medications available for this disease."])

    # Function to display precautions
    def show_precaution():
        predicted_disease = pred1.get()
        precautions = get_precautions(predicted_disease)
        precautions_text = "\n".join(precautions)
        precaution_var.set(precautions_text)

    # Function to display medications
    def show_medication():
        predicted_disease = pred1.get()
        medications = get_medications(predicted_disease)
        medications_text = "\n".join(medications)  # Changed to display each medication on a new line
        medication_var.set(medications_text)

    # Button to predict disease
    tk.Button(root, text="Predict Disease", command=predict, bg="#6495ed", fg="white", font=("Helvetica", 12)).grid(row=6,
                                                                                                                 column=0,
                                                                                                                 padx=10,
                                                                                                                 pady=10,
                                                                                                                 columnspan=2)

    # Create Label for Prediction Result
    tk.Label(root, text="Disease Predicted:", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=7, column=0,
                                                                                                  padx=10, pady=10)
    tk.Entry(root, textvariable=pred1, state="readonly").grid(row=7, column=1, padx=10, pady=10)

    # Button to show precautions
    tk.Button(root, text="Show Precautions", command=show_precaution, bg="#6495ed", fg="white",
           font=("Helvetica", 12)).grid(row=8, column=0, padx=10, pady=10, columnspan=2)

    # Create Label for Precautions
    tk.Label(root, text="Precautions:", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=9, column=0, padx=10,
                                                                                            pady=10)
    tk.Label(root, textvariable=precaution_var, wraplength=400, justify=LEFT, bg="#f0f8ff", font=("Helvetica", 10)).grid(
        row=10, column=0, columnspan=2, padx=10, pady=10)

    # Button to show medications
    tk.Button(root, text="Show Medications", command=show_medication, bg="#6495ed", fg="white",
           font=("Helvetica", 12)).grid(row=11, column=0, padx=10, pady=10, columnspan=2)

    # Create Label for Medications
    tk.Label(root, text="Medications:", bg="#4682b4", fg="white", font=("Helvetica", 12)).grid(row=12, column=0, padx=10,
                                                                                            pady=10)
    tk.Label(root, textvariable=medication_var, wraplength=400, justify=LEFT, bg="#f0f8ff", font=("Helvetica", 10)).grid(
        row=13, column=0, columnspan=2, padx=10, pady=10)

    # Bring the window to the front
    root.lift()
    root.focus_force()

    # Run the Tkinter main loop
    root.mainloop()


if __name__ == "__main__":
    app.run(debug=True)
# -*- coding: utf-8 -*-

