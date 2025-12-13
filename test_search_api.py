"""Test search API response"""
import requests
import json

# Login first
login_response = requests.post('http://localhost:8000/api/auth/login', json={
    'username': 'admin',
    'password': 'admin123'
})

if login_response.status_code == 200:
    token = login_response.json()['access_token']
    
    # Search
    search_response = requests.post(
        'http://localhost:8000/api/search/text',
        json={'query': 'glass roof', 'top_k': 3},
        headers={'Authorization': f'Bearer {token}'}
    )
    
    if search_response.status_code == 200:
        data = search_response.json()
        print(f"Total results: {data['total']}")
        print(f"Duration: {data['duration']:.3f}s\n")
        
        for i, result in enumerate(data['results'][:3], 1):
            print(f"Result {i}:")
            print(f"  Path: {result['path']}")
            print(f"  Thumbnail URL: {result.get('thumbnail_url')}")
            print(f"  Score: {result['score']:.3f}")
            print()
    else:
        print(f"Search failed: {search_response.status_code}")
        print(search_response.text)
else:
    print(f"Login failed: {login_response.status_code}")
