import mysql.connector

#mysql connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="mint"
)

mycursor = mydb.cursor()

def insert_Data(empno,time,name):
    
    time.strftime('%Y-%m-%d %H:%M:%S')
    
    sql = "INSERT INTO hr_allattendance (emp_no, checktime,machine_serial_no) VALUES (%s, %s ,%s)"
    val = (empno,time,name)
    mycursor.execute(sql, val)

    mydb.commit()
    print(mycursor.rowcount, "record inserted.")