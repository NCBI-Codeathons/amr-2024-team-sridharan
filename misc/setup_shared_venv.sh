#!/bin/bash

# Check if the script is run as root or using sudo
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root or using sudo."
  exit 1
fi

# Variables
GROUP_NAME="devgroup"
SHARED_VENV_DIR="/shared_venv"
VENV_NAME="venv"

# Function to add user to the group
add_user_to_group() {
  read -p "Enter the username to add to $GROUP_NAME: " username
  sudo usermod -aG "$GROUP_NAME" "$username"
  echo "User $username added to group $GROUP_NAME."
}

# Create the devgroup if it doesn't already exist
if [ $(getent group "$GROUP_NAME") ]; then
  echo "Group $GROUP_NAME already exists."
else
  echo "Creating group $GROUP_NAME..."
  sudo groupadd "$GROUP_NAME"
  echo "Group $GROUP_NAME created."
fi

# Add users to the group
add_user_to_group
while true; do
  read -p "Do you want to add another user to the group? (y/n): " yn
  case $yn in
    [Yy]* ) add_user_to_group ;;
    [Nn]* ) break ;;
    * ) echo "Please answer yes or no." ;;
  esac
done

# Create the shared virtual environment directory
if [ -d "$SHARED_VENV_DIR" ]; then
  echo "Directory $SHARED_VENV_DIR already exists."
else
  echo "Creating shared directory $SHARED_VENV_DIR..."
  sudo mkdir "$SHARED_VENV_DIR"
  echo "Directory $SHARED_VENV_DIR created."
fi

# Set ownership and permissions for the shared directory
echo "Setting group ownership and permissions for $SHARED_VENV_DIR..."
sudo chown :$GROUP_NAME "$SHARED_VENV_DIR"
sudo chmod 775 "$SHARED_VENV_DIR"

# Navigate to the shared directory and create the virtual environment
cd "$SHARED_VENV_DIR" || exit
if [ -d "$VENV_NAME" ]; then
  echo "Virtual environment $VENV_NAME already exists."
else
  echo "Creating virtual environment in $SHARED_VENV_DIR..."
  python3 -m venv "$VENV_NAME"
  echo "Virtual environment created."
fi

# Set ownership and permissions for the virtual environment
echo "Setting group ownership and permissions for $VENV_NAME..."
sudo chown -R :$GROUP_NAME "$VENV_NAME"
sudo chmod -R 775 "$VENV_NAME"

# Set the setgid bit on the venv directory to preserve group permissions
echo "Setting setgid bit to ensure group permissions for future files..."
sudo chmod g+s "$VENV_NAME"

# Activate the virtual environment
echo "You can now activate the virtual environment by running:"
echo "source $SHARED_VENV_DIR/$VENV_NAME/bin/activate"
