function varargout = splitTheDatastore(datastore,newlabels, optional)

%% Description
%  ------------------------------------------------------------------------
%  A function that splits the datastore into training and testing 
%  datastores based on user's input that decides whether the data should be 
%  processed to have equal number of labels or not. The optional variable 
%  opt gives the user the choice to either return the training and testing 
%  label tables with the count of the former or not.


% Inputs:
% -------------------------------------------------------------------------
% => datastore:     The imageDatastore that contains the raw data.
%
% => newlabels:     A variable that contains the labels that are extracted 
%                   from the initial datastore.
% => optional:      Char or string variable that takes two discrete values: 
%                   "Equal" or "Proceed". Based on one of the two, the 
%                   appropriate measures are taken to treat the splitting 
%                   procedure.

% Outputs: 
%
% => varargout:     As the name depicts, a variable argument output, that
%                   based on flag, the function returns the table of 
%                   training and testing datastores that contain both the 
%                   labels and the count of them for the user to process.
arguments (Input)

    datastore          {mustBeUnderlyingType(datastore,'matlab.io.datastore.ImageDatastore')} 

    newlabels          {mustBeNonempty} 

    optional.flag      {mustBeNumericOrLogical} = false
    
    

end

%% Message for the user and setting the option of label data management.

msg = ['Write "Equal" if you want the number of label of the data in the datastore to be equal in ' ...
       'number, or "Proceed" if you want to continue with the process'];
uiwait(msgbox(msg,'Message'))

% Creation of an input dialog window that takes users input and sets the option variable that 
% handles the method of creating equal number of label in the data, or proceeds with the 
prompt = {'Enter option'};
dlgtitle = 'Option';
fieldsize = [1 45];
definput = {'Equal'};

option = char(inputdlg(prompt,dlgtitle,fieldsize,definput));


if numel(nargin) < 4 
        
    option = validatestring(option,["Equal","Proceed"]);
    
    switch option

        case "Equal"

            fprintf('You chose your data to have equal number of labels\n')
            
            % Find the smallest number of labels of all the categories
            minNum=min(newlabels{:,2});

            % Reduce the original datastore so that all classes have the 
            % same number of labels as the class with the least number of 
            % labels.

            datastore = splitEachLabel(datastore,minNum);
            [Trainds,Testds] = splitEachLabel(datastore,0.7,'randomized');
    
            trainlabelcount_minclass=countEachLabel(Trainds);
            testlabelcount_minclass=countEachLabel(Testds);
            
            if optional.flag

                varargout = {Trainds,Testds,trainlabelcount_minclass, testlabelcount_minclass};

            else

                varargout = {Trainds,Testds};

            end

        case "Proceed"

            % imshow(imread("34yhxs.jpg")); pause(1); close all

            % fprintf('So you have chosen...biases\n')
            [Trainds,Testds] = splitEachLabel(datastore,0.7,'randomized');
            trainlabelcount=countEachLabel(Trainds);
            testlabelcount=countEachLabel(Testds);
            
            
            if optional.flag
                varargout = {Trainds,Testds,trainlabelcount,testlabelcount};
            else
                varargout = {Trainds,Testds};
            end

        otherwise

            error('Unknown option: Την παλεύεις;')

    end

else

    error("Too many function inputs")

end


end