from django.db import models

# Create your models here.

class File_saving_model(models.Model):
    myfile = models.FileField(upload_to='the_app/static/uploaded_files/')
    # renaming file
    def save(self, *args, **kwargs):
        if self.myfile:
            self.myfile.name = "input_file.csv" 
        super().save(*args, **kwargs)
