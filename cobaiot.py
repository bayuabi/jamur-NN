from firebase import firebase
import requests

url = 'https://jamur-ca24c.firebaseio.com/'
firebase = firebase.FirebaseApplication(url,None)

post = firebase.patch('',{'suhu':40})
