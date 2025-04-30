% Step 1: Fetch Codeforces Submissions via API
username = 'Ahmed_Faraj';
api_url = "https://codeforces.com/api/user.status?handle=" + username;
data_json = webread(api_url);

if isfield(data_json, 'result')
    submissions = data_json.result;
else
    error('Error fetching data. Check API response.');
end

num_samples = min(8000, length(submissions));

%% Step 2: Extract Features from Submissions (Improved)
% Preallocate feature arrays (as doubles)
PassedTestCount = zeros(num_samples,1);
TimeConsumedMillis = zeros(num_samples,1);
MemoryConsumedBytes = zeros(num_samples,1);
RelativeTimeSeconds = zeros(num_samples,1);
ProblemRating = zeros(num_samples,1);
% New: Submission Category (0 = Accepted, 1 = Bug, 2 = Inefficiency)
SubCatNumeric = zeros(num_samples,1);

for i = 1:num_samples
    % Handle both cell and struct array cases:
    if iscell(submissions)
        sub = submissions{i};
    else
        sub = submissions(i);
    end
    
    % Extract numeric features; if field is missing or not numeric, use 0 or simulated value.
    if isfield(sub, 'passedTestCount')
        val = sub.passedTestCount;
        if isnumeric(val)
            PassedTestCount(i) = val;
        elseif isstruct(val) && isfield(val, 'data')
            PassedTestCount(i) = double(val.data);
        else
            PassedTestCount(i) = 0;
        end
    else
        PassedTestCount(i) = 0;
    end
    
    if isfield(sub, 'timeConsumedMillis')
        val = sub.timeConsumedMillis;
        if isnumeric(val)
            TimeConsumedMillis(i) = val;
        elseif isstruct(val) && isfield(val, 'data')
            TimeConsumedMillis(i) = double(val.data);
        else
            TimeConsumedMillis(i) = 0;
        end
    else
        TimeConsumedMillis(i) = 0;
    end
    
    if isfield(sub, 'memoryConsumedBytes')
        val = sub.memoryConsumedBytes;
        if isnumeric(val)
            MemoryConsumedBytes(i) = val;
        elseif isstruct(val) && isfield(val, 'data')
            MemoryConsumedBytes(i) = double(val.data);
        else
            MemoryConsumedBytes(i) = 0;
        end
    else
        MemoryConsumedBytes(i) = 0;
    end
    
    if isfield(sub, 'relativeTimeSeconds')
        val = sub.relativeTimeSeconds;
        if isnumeric(val)
            RelativeTimeSeconds(i) = val;
        elseif isstruct(val) && isfield(val, 'data')
            RelativeTimeSeconds(i) = double(val.data);
        else
            RelativeTimeSeconds(i) = 0;
        end
    else
        RelativeTimeSeconds(i) = 0;
    end
    
    % Problem rating: if available, convert to double; otherwise, simulate a rating.
    if isfield(sub, 'problem') && isfield(sub.problem, 'rating')
        val = sub.problem.rating;
        if isnumeric(val)
            ProblemRating(i) = val;
        elseif isstruct(val) && isfield(val, 'data')
            ProblemRating(i) = double(val.data);
        else
            ProblemRating(i) = randi([800,3500]);
        end
    else
        ProblemRating(i) = randi([800,3500]);
    end
    
    % Define Submission Category:
    % 0: Accepted, 1: Bug (WRONG_ANSWER or RUNTIME_ERROR),
    % 2: Inefficiency (TIME_LIMIT_EXCEEDED or MEMORY_LIMIT_EXCEEDED)
    if isfield(sub, 'verdict')
        verdict = sub.verdict;
        if strcmp(verdict, 'WRONG_ANSWER') || strcmp(verdict, 'RUNTIME_ERROR')
            SubCatNumeric(i) = 1;
        elseif strcmp(verdict, 'TIME_LIMIT_EXCEEDED') || strcmp(verdict, 'MEMORY_LIMIT_EXCEEDED')
            SubCatNumeric(i) = 2;
        else
            SubCatNumeric(i) = 0;
        end
    else
        SubCatNumeric(i) = 0;
    end
end

% Convert numeric labels to categorical variable
SubmissionCategory = categorical(SubCatNumeric, [0,1,2], {'Accepted','Bug','Inefficiency'});

%% Step 3: Create Dataset
data = table(PassedTestCount, TimeConsumedMillis, MemoryConsumedBytes, RelativeTimeSeconds, ProblemRating, SubmissionCategory, ...
    'VariableNames', {'PassedTestCount', 'TimeConsumedMillis', 'MemoryConsumedBytes', 'RelativeTimeSeconds', 'ProblemRating', 'SubmissionCategory'});

%% (Optional) Debug: Display Unique Programming Languages (if needed)
if iscell(submissions)
    langs = cellfun(@(s) s.programmingLanguage, submissions, 'UniformOutput', false);
else
    langs = {submissions.programmingLanguage};
end
disp('Unique programming languages:');
disp(unique(langs));

%% Step 4: Balance the Dataset
cats = categories(data.SubmissionCategory);
counts = countcats(data.SubmissionCategory);
min_samples_bal = min(counts);

if min_samples_bal < 20
    warning('Not enough data in one or more categories. Using all data instead.');
    data_balanced = data;
else
    data_balanced = [];
    for j = 1:numel(cats)
        idx = data.SubmissionCategory == cats{j};
        catData = data(idx, :);
        catBalanced = datasample(catData, min_samples_bal, 'Replace', false);
        data_balanced = [data_balanced; catBalanced]; %#ok<AGROW>
    end
end

%% Step 5: Set Up Cross-Validation or Hold-Out Split
n = height(data_balanced);
if n < 2
    error('Not enough samples after balancing to perform cross-validation. Try increasing num_samples.');
end

if n < 10
    warning('Not enough data for k-fold CV. Using hold-out split (50%%) instead.');
    cv = cvpartition(n, 'HoldOut', 0.5, 'Stratify', false);
    k = 1;
else
    k = 5;
    if n < k
        k = n;
        warning('Reduced number of folds to %d due to small dataset size.', k);
    end
    try
        cv = cvpartition(n, 'KFold', k, 'Stratify', true);
    catch ME
        warning(ME.identifier, '%s. Using non-stratified CV instead.', ME.message);
        cv = cvpartition(n, 'KFold', k, 'Stratify', false);
    end
end

% For multi-class, determine number of categories:
numCategories = numel(categories(data_balanced.SubmissionCategory));
total_cm_dt = zeros(numCategories, numCategories);
total_cm_rf = zeros(numCategories, numCategories);
total_cm_knn = zeros(numCategories, numCategories);
total_cm_svm = zeros(numCategories, numCategories);

accuracy_dt = zeros(k, 1);
accuracy_rf = zeros(k, 1);
accuracy_knn = zeros(k, 1);
accuracy_svm = zeros(k, 1);

%% Step 6: Perform Cross-Validation with Feature Scaling
for i = 1:k
    if k == 1
        trainData = data_balanced(training(cv), :);
        testData  = data_balanced(test(cv), :);
    else
        trainData = data_balanced(training(cv, i), :);
        testData = data_balanced(test(cv, i), :);
    end
    
    % Extract predictors (all columns except SubmissionCategory) and response.
    X_train = trainData{:, 1:end-1};
    Y_train = trainData.SubmissionCategory;
    X_test = testData{:, 1:end-1};
    Y_test = testData.SubmissionCategory;
    
    % Normalize predictors using z-score (based on training data)
    [X_train, mu, sigma] = zscore(X_train);
    X_test = (X_test - mu) ./ sigma;
    
    %% Train Decision Tree Model with Pruning
    dt_model = fitctree(X_train, Y_train, 'MinLeafSize', 10, 'MaxNumSplits', 10);
    
    %% Train Random Forest Model
    dt_template = templateTree('MaxNumSplits', 10);
    rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 150, 'Learners', dt_template);
    
    %% Train KNN Model with k=7
    knn_model = fitcknn(X_train, Y_train, 'NumNeighbors', 7);
    
    %% Train SVM Model using fitcecoc for multi-class classification
    svm_model = fitcecoc(X_train, Y_train);
    
    %% Evaluate Models on Test Set
    pred_dt = predict(dt_model, X_test);
    pred_rf = predict(rf_model, X_test);
    pred_knn = predict(knn_model, X_test);
    pred_svm = predict(svm_model, X_test);
    
    % Compute confusion matrices for each model
    cm_dt = confusionmat(Y_test, pred_dt);
    cm_rf = confusionmat(Y_test, pred_rf);
    cm_knn = confusionmat(Y_test, pred_knn);
    cm_svm = confusionmat(Y_test, pred_svm);
    
    % Ensure each confusion matrix is of size numCategories x numCategories
    if size(cm_dt,1) < numCategories, cm_dt(numCategories,numCategories) = 0; end
    if size(cm_rf,1) < numCategories, cm_rf(numCategories,numCategories) = 0; end
    if size(cm_knn,1) < numCategories, cm_knn(numCategories,numCategories) = 0; end
    if size(cm_svm,1) < numCategories, cm_svm(numCategories,numCategories) = 0; end
    
    % Accumulate confusion matrices
    total_cm_dt = total_cm_dt + cm_dt;
    total_cm_rf = total_cm_rf + cm_rf;
    total_cm_knn = total_cm_knn + cm_knn;
    total_cm_svm = total_cm_svm + cm_svm;
    
    % Calculate accuracy for current fold
    accuracy_dt(i) = sum(diag(cm_dt)) / sum(cm_dt(:));
    accuracy_rf(i) = sum(diag(cm_rf)) / sum(cm_rf(:));
    accuracy_knn(i) = sum(diag(cm_knn)) / sum(cm_knn(:));
    accuracy_svm(i) = sum(diag(cm_svm)) / sum(cm_svm(:));
end

%% Step 7: Display Average Accuracy Across Folds
accuracy_results = table({'Decision Tree'; 'Random Forest'; 'KNN'; 'SVM'}, ...
                         [mean(accuracy_dt); mean(accuracy_rf); mean(accuracy_knn); mean(accuracy_svm)], ...
                         'VariableNames', {'Model', 'Accuracy'});
disp(accuracy_results);

%% Step 8: Visualizations and Saving Figures

% Figure 1: Scatter Plot (PassedTestCount vs. TimeConsumedMillis)
figure;
gscatter(data_balanced{:, 'PassedTestCount'}, data_balanced{:, 'TimeConsumedMillis'}, data_balanced{:, 'SubmissionCategory'}, 'rbg', 'o');
title('Passed Test Count vs Time Consumed (ms) Colored by Submission Category');
xlabel('Passed Test Count'); ylabel('Time Consumed (ms)');
saveas(gcf, 'scatter_passed_vs_time.png');

% Figure 2: Histogram of Problem Ratings
figure;
histogram(data_balanced{:, 'ProblemRating'});
title('Distribution of Problem Ratings');
xlabel('Problem Rating'); ylabel('Frequency');
saveas(gcf, 'histogram_problem_ratings.png');

% Figure 3: Aggregated Confusion Matrix for Decision Tree
figure;
confusionchart(total_cm_dt, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Aggregated Confusion Matrix: Decision Tree');
saveas(gcf, 'confusion_matrix_decision_tree.png');

% Figure 4: Aggregated Confusion Matrix for Random Forest
figure;
confusionchart(total_cm_rf, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Aggregated Confusion Matrix: Random Forest');
saveas(gcf, 'confusion_matrix_random_forest.png');

% Figure 5: Aggregated Confusion Matrix for KNN
figure;
confusionchart(total_cm_knn, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Aggregated Confusion Matrix: KNN');
saveas(gcf, 'confusion_matrix_knn.png');

% Figure 6: Aggregated Confusion Matrix for SVM
figure;
confusionchart(total_cm_svm, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Aggregated Confusion Matrix: SVM');
saveas(gcf, 'confusion_matrix_svm.png');
