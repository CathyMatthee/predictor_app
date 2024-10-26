# Use the official Python image as a base
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any necessary packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

EXPOSE 5000

# Command to run your application
ENTRYPOINT ["python"]
CMD ["app.py"]

