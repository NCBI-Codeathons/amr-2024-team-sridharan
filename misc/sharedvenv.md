# Shared Virtual Environment Setup for Multiple Users on GCP VM

This guide provides instructions on how to set up a shared Python virtual environment on a Google Cloud Platform (GCP) Virtual Machine (VM) that can be accessed and used by multiple users. We will create a shared group, assign users to that group, and configure the appropriate permissions so that users can collaboratively use the virtual environment.

## Prerequisites

- You need administrative (sudo) access on the GCP VM.
- Python 3 must be installed on the VM.

## Steps to Set Up Shared Virtual Environment

### 1. Create a User Group

Create a new group called `devgroup`, which will allow multiple users to share access to the virtual environment.

```bash
sudo groupadd devgroup
```

### 2. Add Users to the Group

Add each user that will collaborate on the VM to the newly created `devgroup`. Replace `<username>` with the actual username of each user.

```bash
sudo usermod -aG devgroup <username>
```

Repeat this step for each user that needs access to the shared environment.

### 3. Create a Shared Directory for the Virtual Environment

Create a directory where the shared virtual environment will reside.

```bash
sudo mkdir /shared_venv
```

### 4. Assign Group Ownership to the Directory

Change the group ownership of the shared directory to `devgroup`.

```bash
sudo chown :devgroup /shared_venv
```

### 5. Set Permissions on the Shared Directory

Ensure that the group has read, write, and execute permissions on the shared directory.

```bash
sudo chmod 775 /shared_venv
```

### 6. Create the Virtual Environment

Navigate to the shared directory and create a Python virtual environment.

```bash
cd /shared_venv
python3 -m venv venv
```

### 7. Assign Group Ownership and Set Permissions on the Virtual Environment

Ensure that the `venv` directory (the virtual environment) has the correct group ownership and permissions so that all users in `devgroup` can access it.

```bash
sudo chown -R :devgroup venv
sudo chmod -R 775 venv
```

### 8. Activate the Virtual Environment

Each user can now activate the shared virtual environment by running:

```bash
source /shared_venv/venv/bin/activate
```

### 9. Preserve Group Permissions for Future Files and Directories

To ensure that any files or directories created within the `venv` directory inherit the correct group permissions, set the "setgid" bit on the `venv` directory.

```bash
sudo chmod g+s /shared_venv/venv
```

## Usage

Once the setup is complete, all users who are part of the `devgroup` can:

1. SSH into the VM.
2. Activate the shared virtual environment using:

   ```bash
   source /shared_venv/venv/bin/activate
   ```

3. Install Python packages or run Python scripts in the shared environment. All changes (e.g., new package installations) will be available to all users.

## Notes

- Make sure each user has the correct permissions to access and modify files within the shared environment.
- If any new users are added to the VM later, remember to add them to the `devgroup` and ensure they have access to the shared virtual environment directory. 

