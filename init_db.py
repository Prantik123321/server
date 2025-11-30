from app import app, db
from models import User

def init_database():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create admin user for testing
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@game.com')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Database initialized with admin user!")
        else:
            print("Database already initialized!")

if __name__ == '__main__':
    init_database()