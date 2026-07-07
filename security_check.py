import urllib.request
import urllib.error
import json
import subprocess
import sys
import ssl

def get_packages():
    # Attempt to use pip freeze to get packages
    try:
        result = subprocess.run([".\\env\\Scripts\\pip", "freeze"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
    except Exception as e:
        print(f"Failed to run pip freeze: {e}")
        return []
    
    packages = []
    for line in lines:
        if "==" in line:
            name, version = line.split("==")
            packages.append((name.strip(), version.strip()))
    return packages

def check_osv(name, version):
    url = "https://api.osv.dev/v1/query"
    data = json.dumps({
        "version": version,
        "package": {"name": name, "ecosystem": "PyPI"}
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    # Bypass SSL verification if needed, although mostly fine
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("vulns", [])
    except Exception as e:
        print(f"Error checking {name}=={version}: {e}")
        return None

def main():
    packages = get_packages()
    if not packages:
        print("No packages found or failed to execute pip freeze.")
        sys.exit(1)
        
    print(f"Checking {len(packages)} packages against OSV database...")
    found_vulns = 0
    for name, version in packages:
        vulns = check_osv(name, version)
        if vulns:
            print(f"\n[!] Vulnerability found in {name}=={version}:")
            for v in vulns:
                aliases = v.get("aliases", [])
                print(f"  - ID: {v.get('id')} / Aliases: {', '.join(aliases)}")
                details = v.get("details", "No details provided").replace("\n", " ")
                print(f"    Details: {details[:150]}...")
            found_vulns += 1
            
    if found_vulns == 0:
        print("\nNo known vulnerabilities found in any packages.")
    else:
        print(f"\nFound vulnerabilities in {found_vulns} packages.")

if __name__ == "__main__":
    main()
