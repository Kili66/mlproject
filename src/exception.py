import sys


#whenever an exception is error is detected, find the error message 
def error_message_detail(error, error_detail:sys):
    
    #take the last error information and put it in exc_tb variable
    _,_,exc_tb=error_detail.exc_info()
    #find the file name
    file_name= exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        #find the error, the error file name and the error line
        file_name, exc_tb.tb_lineno, str(error))
        
    return error_message
   
    
#This class inherit Exeception
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
    #print error message here
    def __str__(self):
        return self.error_message