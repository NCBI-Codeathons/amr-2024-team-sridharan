### Instructions to Use the Script

1. **Save the script:**
   - Save the above script as `setup_shared_venv.sh`.

2. **Make the script executable:**
   - Run the following command to give the script execution permissions:

     ```bash
     chmod +x setup_shared_venv.sh
     ```

3. **Run the script:**
   - Execute the script with sudo:

     ```bash
     sudo ./setup_shared_venv.sh
     ```

   The script will:
   - Check if you are running it as root or with sudo.
   - Create the `devgroup` if it doesn't exist.
   - Add users to the group interactively.
   - Create the shared virtual environment directory and set appropriate permissions.
   - Create a virtual environment inside the shared directory.
   - Set up group permissions for future file creation in the virtual environment directory.
   - Output the command to activate the virtual environment.

This reusable script simplifies the process and ensures that it can be repeated whenever needed, with minimal user intervention.
