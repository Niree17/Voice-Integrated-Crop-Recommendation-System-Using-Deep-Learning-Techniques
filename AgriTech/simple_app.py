# Function to make prediction using Groq API
def predict_with_groq(input_data, model_name):
    """Use Groq API to predict crop based on input parameters"""
    # Format the input data for the prompt
    ph, n, k20, temperature, rainfall = input_data
    
    prompt = f"""
    As an agricultural expert, predict the most suitable crop based on these soil and climate parameters:
    
    - pH: {ph}
    - Nitrogen (N): {n} kg/ha
    - Potassium (K2O): {k20} kg/ha
    - Temperature: {temperature} °C
    - Rainfall: {rainfall} mm
    
    Based on these parameters, what is the most suitable crop to grow? 
    Provide only the crop name and a confidence percentage (0-100).
    Format your response exactly like this example: "paddy,85"
    """
    
    # Call Groq API
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=100,
            top_p=1,
            stream=False,
        )
        
        # Extract the response
        response = chat_completion.choices[0].message.content.strip()
        
        # Parse the response (expected format: "crop_name,confidence")
        try:
            crop, confidence_str = response.split(',')
            confidence = float(confidence_str)
        except:
            # Fallback if response format is unexpected
            crop = response[:20]  # Take first 20 chars as crop name
            confidence = 70.0     # Default confidence
            
        return crop.strip(), confidence
        
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        # Fallback to a default prediction
        return "paddy", 60.0