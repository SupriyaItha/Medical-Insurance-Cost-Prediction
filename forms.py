from django import forms

class File_saving_form(forms.Form):
    myfile = forms.FileField()
