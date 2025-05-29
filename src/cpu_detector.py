# windows_cpu_detector.py
import platform
import subprocess
import os
import re
import winreg

def get_cpu_friendly_name():
    """Get a user-friendly CPU name on Windows systems."""
    
    # Method 1: Try PowerShell with full path and better error handling
    try:
        # Use the full path to PowerShell
        powershell_path = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 
                                     r'System32\WindowsPowerShell\v1.0\powershell.exe')
        
        if os.path.exists(powershell_path):
            cmd = [powershell_path, "-Command", 
                  "(Get-CimInstance -ClassName Win32_Processor).Name"]
            result = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
            if result.strip():
                return result.strip()
    except Exception as e:
        print(f"PowerShell method failed: {e}")
    
    # Method 2: Try Registry query for ProcessorNameString
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        winreg.CloseKey(key)
        if cpu_name and cpu_name.strip():
            return cpu_name.strip()
    except Exception as e:
        print(f"Registry method failed: {e}")
    
    # Method 3: Parse platform.processor() to make it more readable
    processor_info = platform.processor()
    
    # Check if it's an AMD processor
    if "AuthenticAMD" in processor_info:
        # Try to extract family/model info and map to Ryzen naming
        family_match = re.search(r"Family (\d+)", processor_info)
        model_match = re.search(r"Model (\d+)", processor_info)
        
        if family_match and model_match:
            family = int(family_match.group(1))
            model = int(model_match.group(1))
            
            # Map to Ryzen names based on family/model numbers
            # This is a simplified mapping and may not be 100% accurate
            if family == 26:  # Zen 5 architecture
                return "AMD Ryzen 9000 Series"
            elif family == 25:  # Zen 4 architecture
                return "AMD Ryzen 7000 Series"
            elif family == 24:
                return "AMD Ryzen (Post-Zen3)"
            elif family == 23:
                return "AMD Ryzen (Zen2/Zen3)"
            
        return "AMD Ryzen Processor"
    
    # For Intel processors
    elif "GenuineIntel" in processor_info:
        return "Intel Processor"
    
    # Fallback to cleaned up platform.processor()
    clean_processor = processor_info.replace("Family", "").replace("Model", "").replace(
        "Stepping", "").replace("AuthenticAMD", "AMD").replace("GenuineIntel", "Intel")
    clean_processor = re.sub(r'\s+', ' ', clean_processor).strip()
    
    return clean_processor or "CPU"

# Add this function to clean up the CPU name
def clean_cpu_name_for_ui(cpu_name):
    """Clean up CPU name for UI display by removing unnecessary words."""
    # Remove generic terms
    for term in ["Processor", "Core", "CPU", "Detected"]:
        cpu_name = cpu_name.replace(f" {term}", "")
    
    # Special handling for common CPU patterns
    if "AMD Ryzen" in cpu_name:
        # Extract just "AMD Ryzen X XXXX" part
        import re
        match = re.search(r"(AMD Ryzen \d+ \d+X+)", cpu_name)
        if match:
            return match.group(1)
        
        # Another pattern for Ryzen
        match = re.search(r"(AMD Ryzen \d+ \d+)", cpu_name)
        if match:
            return match.group(1)
    
    # For Intel processors
    if "Intel" in cpu_name and "GHz" in cpu_name:
        # Remove frequency info
        cpu_name = cpu_name.split("@")[0].strip()
    
    # Remove any extra spaces
    import re
    cpu_name = re.sub(r'\s+', ' ', cpu_name).strip()
    
    return cpu_name

if __name__ == "__main__":
    print("Windows CPU Detection Test")
    print("-------------------------")
    print(f"Platform: {platform.platform()}")
    print(f"Raw processor info: {platform.processor()}")
    print(f"\nDetected CPU: {get_cpu_friendly_name()}")