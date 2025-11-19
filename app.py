from flask import Flask, render_template

# Create the Flask application instance
app = Flask(__name__)

# Define the route for the main page (the root URL '/')
@app.route('/')
def hello_world():
    # Render the HTML template named 'index.html'
    return render_template('index.html')

# This allows the script to be run directly
if __name__ == '__main__':
    # Run the application in debug mode for easy testing
    app.run(debug=True)