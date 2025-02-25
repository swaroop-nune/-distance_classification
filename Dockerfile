# Use the official Python image as a base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install required Python dependencies
RUN pip install numpy pandas scikit-learn wandb

# Command to run your script
CMD ["python", "distance_classification.py"]
