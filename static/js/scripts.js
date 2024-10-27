

// Pre-poluate the predictor app with fields from dataset 1 or 2 in dropdown
function populateFields(selectedID) {
    d3.json("http://127.0.0.1:5000/api_json_connectivity").then((data) => {
      console.log(data);
      console.log(selectedID)
  

      // Filter with fat arrow method the data for the object with the desired sample number
      let selectedItem = data.find(meta => meta.id == selectedID);
          console.log(selectedItem); // prints patient data for selected ID)

            // Define the desired order of fields as per your HTML structure
            let orderedFields = [
                "area_mean", 
                "area_worst", 
                "compactness_mean", 
                "compactness_se", 
                "compactness_worst", 
                "concave points_mean", 
                "concave points_se", 
                "concave points_worst", 
                "concavity_mean", 
                "concavity_worst", 
                "fractal_dimension_mean", 
                "fractal_dimension_worst", 
                "perimeter_mean", 
                "perimeter_worst", 
                "radius_mean", 
                "radius_worst", 
                "smoothness_mean", 
                "smoothness_se", 
                "smoothness_worst", 
                "symmetry_mean", 
                "symmetry_se", 
                "symmetry_worst", 
                "texture_mean", 
                "texture_se", 
                "texture_worst", 
                "area_se", 
                "concavity_se", 
                "perimeter_se", 
                "radius_se", 
                "fractal_dimension_se"
            ];
      
        // Populate input fields
        orderedFields.forEach(field => {
                d3.select(`input[name="${field}"]`).property("value", selectedItem[field]);
            }
        );

    });    
}

  // Function to clear all fields
function clearFields() {
    document.querySelectorAll('input[type="text"]').forEach(input => {
        input.value = ''; // Clear each input field
    });
}
  
  // Function to run on page load - populate the dropdown menu with demo data
  function init() {
    d3.json("http://127.0.0.1:5000/api_json_connectivity").then((data) => {
        console.log("Fetched data on init:", data); // Log the fetched data

      // Use d3 to populate and select the dropdown with id of demo data
      let dropdownMenu = d3.select("#dataSelection");
  
      // Use ids to populate the dropdown options
      // Hint: Inside a loop, use d3 to append the last 2 lines of data
      for (let i = data.length-2; i < data.length; i++) { //looping over array
        dropdownMenu.append("option")
            .text(`ID: ${data[i].id} - Diagnosis: ${data[i].diagnosis}`)
      };
    });
  }
  
    // Function for event listener
    function optionChanged(selectedOption) {
      // Extract the ID from the selected option
      let selectedID = selectedOption.split(" - ")[0].split(": ")[1];
      populateFields(selectedID);
    }
  
  // Initialise the dashboard
  init();