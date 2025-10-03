import sqlite3


def create_connection(db_file):
    # Connect to sqlite database file
    conn = sqlite3.connect(db_file)
    return conn

def execute_sql(conn, sql):
    # Cursor makes connection to database
    c = conn.cursor()
    # Execute sql query
    c.execute(sql)
    # Commit changes
    conn.commit()
    return None

def main():
    database = "sqlite.db"
    sql_file = 'sql/table.sql'
    # Read sql from txt file
    fd = open(sql_file, 'r')
    sql_file = fd.read()
    fd.close()
    # Generate 40 columns programatically
    latent_vectors = ', '.join([f'latent_vector_{i} float' for i in range(40)]) + ','

    sql_file = sql_file.format(latent_vectors)
    # Split commands
    sql_commands = sql_file.split(';')
    # coNNECt to database
    conn = create_connection(database)

    for command in sql_commands:
        # Execute each command sequentially
        execute_sql(conn, command)
    
    # Close connection
    conn.close()

   

if __name__ == '__main__':
    main()