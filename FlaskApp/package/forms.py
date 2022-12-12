from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired

class UserForm(FlaskForm):
    userid = IntegerField('UserId', validators=[DataRequired()])
    submit = SubmitField('Submit')