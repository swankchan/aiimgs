"""Test library API"""
import requests

# Login
login_response = requests.post('http://localhost:8000/api/auth/login', json={
    'username': 'admin',
    'password': 'admin123'
})

if login_response.status_code == 200:
    token = login_response.json()['access_token']
    
    # Get library
    library_response = requests.get(
        'http://localhost:8000/api/library/',
        headers={'Authorization': f'Bearer {token}'}
    )
    
    if library_response.status_code == 200:
        data = library_response.json()
        print(f"Total items: {data['total']}")
        print(f"Page: {data['page']} of {data['total_pages']}")
        print(f"Items in response: {len(data['items'])}\n")
        
        for i, item in enumerate(data['items'][:5], 1):
            print(f"Item {i}:")
            print(f"  Path: {item['path']}")
            print(f"  Type: {item['type']}")
            print(f"  Name: {item['name']}")
            print(f"  Thumbnail: {item.get('thumbnail_url')}")
            print()
    else:
        print(f"Library failed: {library_response.status_code}")
        print(library_response.text)
else:
    print(f"Login failed: {login_response.status_code}")
