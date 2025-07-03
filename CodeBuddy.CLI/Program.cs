using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using OllamaSharp;
using OllamaSharp.Models;
using OllamaSharp.Models.Chat;
using System.Collections;
using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Text;

public class CodeBuddy
{
    private static OllamaApiClient _ollama;
    private static string _modelName = "llama3.1:latest";
    private static string _embeddingModel = "nomic-embed-text:latest";
    private static VectorStore _vectorStore;

    public static async Task Main(string[] args)
    {
        Console.InputEncoding = Encoding.UTF8;
        Console.OutputEncoding = Encoding.UTF8;

        Console.Title = "👾 CodeBuddy - AI Помощник Программиста";
        Console.WriteLine("=== CodeBuddy v1.0 - Ваш Ассистент по Кодовой Базе ===");
        Console.WriteLine("Использует OmniSharp для анализа кода и Ollama для RAG\n");

        // Инициализация Ollama
        try
        {
            _ollama = new OllamaApiClient(new Uri("http://localhost:11434"));
            bool isRunning = await _ollama.IsRunningAsync();
            if (!isRunning) throw new Exception("Сервер Ollama не запущен");

            Console.WriteLine("✅ Подключено к серверу Ollama");
            await EnsureModelsAvailable();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ Ошибка подключения к Ollama: {ex.Message}");
            Console.WriteLine("Убедитесь, что Ollama установлен и запущен (https://ollama.com/)");
            Console.ResetColor();
            return;
        }

        // Загрузка проекта
        Console.WriteLine("\nВведите путь к решению или каталогу проекта:");
        var path = Console.ReadLine()?.Trim('"');

        if (!Directory.Exists(path))
        {
            Console.WriteLine("Каталог не найден.");
            return;
        }

        // Извлечение информации о коде
        Console.WriteLine("\n🔍 Анализ кодовой базы с LINQ-провайдером...");
        var codeFragments = await ExtractCodeInformationAsync(path);
        Console.WriteLine($"📊 Извлечено {codeFragments.Count} фрагментов кода");

        // Построение векторного хранилища
        Console.WriteLine("🧠 Построение базы знаний...");
        _vectorStore = new VectorStore(_ollama, _embeddingModel);
        await _vectorStore.BuildStoreAsync(codeFragments);

        // Основной цикл взаимодействия
        Console.WriteLine("\n💬 CodeBuddy готов к работе. Задавайте вопросы о кодовой базе (введите '/exit' для выхода)");
        Console.WriteLine("==============================================================");

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("\n🧑🏻‍💻 Вы: ");
            Console.ResetColor();

            var question = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(question)) continue;
            if (question.Equals("/exit", StringComparison.OrdinalIgnoreCase)) break;

            // Получение релевантного контекста
            var context = await _vectorStore.SearchAsync(question, topK: 5);

            // Генерация ответа
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("\n🧠 CodeBuddy: ");
            Console.ResetColor();

            var response = await GenerateAnswerAsync(question, context);
            Console.WriteLine(response);
        }
    }

    private static async Task EnsureModelsAvailable()
    {
        var models = (await _ollama.ListLocalModelsAsync()).ToList();

        if (!models.Any(m => m.Name.Contains(_modelName)))
        {
            throw new Exception($"{_modelName} не найден");
        }

        if (!models.Any(m => m.Name.Contains(_embeddingModel)))
        {
            throw new Exception($"{_embeddingModel} не найден");

        }
    }

    private static async Task<List<CodeFragment>> ExtractCodeInformationAsync(string path)
    {
        var fragments = new List<CodeFragment>();

        using var provider = new SourceCodeQueryProvider(path);

        // Извлечение классов
        var classes = new SourceCodeQuery<ClassDeclarationSyntax>(provider).ToList();
        foreach (var classDecl in classes)
        {
            fragments.Add(new CodeFragment(
                Id: Guid.NewGuid().ToString(),
                Content: FormatClassInfo(classDecl),
                Description: $"Класс: {classDecl.Identifier}"
            ));

            // Извлечение методов класса
            var methods = new SourceCodeQuery<MethodDeclarationSyntax>(provider)
                .Where(m => m.Parent == classDecl)
                .ToList();

            foreach (var method in methods)
            {
                fragments.Add(new CodeFragment(
                    Id: Guid.NewGuid().ToString(),
                    Content: FormatMethodInfo(method),
                    Description: $"Метод: {method.Identifier} в {classDecl.Identifier}"
                ));
            }
        }

        // Извлечение интерфейсов
        var interfaces = new SourceCodeQuery<InterfaceDeclarationSyntax>(provider).ToList();
        foreach (var interfaceDecl in interfaces)
        {
            fragments.Add(new CodeFragment(
                Id: Guid.NewGuid().ToString(),
                Content: FormatInterfaceInfo(interfaceDecl),
                Description: $"Интерфейс: {interfaceDecl.Identifier}"
            ));
        }

        // Извлечение пространств имен
        var namespaces = new SourceCodeQuery<NamespaceDeclarationSyntax>(provider).ToList();
        foreach (var ns in namespaces)
        {
            fragments.Add(new CodeFragment(
                Id: Guid.NewGuid().ToString(),
                Content: FormatNamespaceInfo(ns),
                Description: $"Пространство имен: {ns.Name}"
            ));
        }

        // Извлечение файловых пространств имен
        var fileScopedNamespaces = new SourceCodeQuery<FileScopedNamespaceDeclarationSyntax>(provider).ToList();
        foreach (var ns in fileScopedNamespaces)
        {
            fragments.Add(new CodeFragment(
                Id: Guid.NewGuid().ToString(),
                Content: FormatFileScopedNamespaceInfo(ns),
                Description: $"Файловое пространство имен: {ns.Name}"
            ));
        }

        return fragments;
    }

    private static string FormatClassInfo(ClassDeclarationSyntax classDecl)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"КЛАСС: {classDecl.Identifier}");
        sb.AppendLine($"Модификаторы: {string.Join(" ", classDecl.Modifiers)}");

        if (classDecl.BaseList != null)
            sb.AppendLine($"Базовые типы: {string.Join(", ", classDecl.BaseList.Types)}");

        sb.AppendLine("\nЧлены класса:");
        foreach (var member in classDecl.Members)
        {
            if (member is PropertyDeclarationSyntax prop)
                sb.AppendLine($"- Свойство: {prop.Type} {prop.Identifier}");
            else if (member is FieldDeclarationSyntax field)
                sb.AppendLine($"- Поле: {field.Declaration.Variables.First().Identifier}");
        }

        sb.AppendLine("\nДокументация:");
        sb.AppendLine(GetDocumentationComment(classDecl));

        return sb.ToString();
    }

    private static string FormatMethodInfo(MethodDeclarationSyntax method)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"МЕТОД: {method.Identifier}");
        sb.AppendLine($"Возвращаемый тип: {method.ReturnType}");
        sb.AppendLine($"Параметры: {string.Join(", ", method.ParameterList.Parameters)}");
        sb.AppendLine($"Модификаторы: {string.Join(" ", method.Modifiers)}");

        sb.AppendLine("\nДокументация:");
        sb.AppendLine(GetDocumentationComment(method));

        sb.AppendLine("\nРеализация:");
        sb.AppendLine(method.Body?.ToFullString() ?? method.ExpressionBody?.ToFullString() ?? "Нет реализации");

        return sb.ToString();
    }

    private static string FormatInterfaceInfo(InterfaceDeclarationSyntax interfaceDecl)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"ИНТЕРФЕЙС: {interfaceDecl.Identifier}");
        sb.AppendLine($"Модификаторы: {string.Join(" ", interfaceDecl.Modifiers)}");

        if (interfaceDecl.BaseList != null)
            sb.AppendLine($"Базовые интерфейсы: {string.Join(", ", interfaceDecl.BaseList.Types)}");

        sb.AppendLine("\nЧлены интерфейса:");
        foreach (var member in interfaceDecl.Members)
        {
            if (member is MethodDeclarationSyntax method)
                sb.AppendLine($"- Метод: {method.ReturnType} {method.Identifier}");
            else if (member is PropertyDeclarationSyntax prop)
                sb.AppendLine($"- Свойство: {prop.Type} {prop.Identifier}");
        }

        sb.AppendLine("\nДокументация:");
        sb.AppendLine(GetDocumentationComment(interfaceDecl));

        return sb.ToString();
    }

    private static string FormatNamespaceInfo(NamespaceDeclarationSyntax ns)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"ПРОСТРАНСТВО ИМЕН: {ns.Name}");
        sb.AppendLine($"Модификаторы: {string.Join(" ", ns.Modifiers)}");

        sb.AppendLine("\nДокументация:");
        sb.AppendLine(GetDocumentationComment(ns));

        return sb.ToString();
    }

    private static string FormatFileScopedNamespaceInfo(FileScopedNamespaceDeclarationSyntax ns)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"ФАЙЛОВОЕ ПРОСТРАНСТВО ИМЕН: {ns.Name}");
        sb.AppendLine($"Модификаторы: {string.Join(" ", ns.Modifiers)}");

        sb.AppendLine("\nДокументация:");
        sb.AppendLine(GetDocumentationComment(ns));

        return sb.ToString();
    }

    private static string GetDocumentationComment(SyntaxNode node)
    {
        var trivia = node.GetLeadingTrivia();
        var docComments = trivia
            .Where(t => t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) ||
                        t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia))
            .Select(t => t.ToFullString().Trim());

        return string.Join("\n", docComments) ?? "Нет документации";
    }

    private static async Task<string> GenerateAnswerAsync(string question, string context)
    {
        var prompt = $"""
            [КОНТЕКСТ]
            {context}
            
            [ВОПРОС]
            {question}
            
            [ИНСТРУКЦИИ]
            1. Отвечайте ТОЛЬКО на основе контекста
            2. Будьте технически точны и кратки
            3. Если не уверены, предложите где искать в кодовой базе
            """;

        var request = new ChatRequest
        {
            Model = _modelName,
            Messages =
            [
                new(ChatRole.System, "Вы CodeBuddy, AI-ассистент для .NET разработчиков. Отвечайте на вопросы о кодовой базе, используя предоставленный контекст."),
                new(ChatRole.User, prompt)
            ],
            Stream = false
        };

        StringBuilder responce = new StringBuilder();

        await foreach (var r in _ollama.ChatAsync(request, CancellationToken.None))
        {
            Console.Write(responce.AppendLine(r.Message.Content));
        }

        return responce.ToString();
    }
}

// Класс для хранения фрагментов кода
public record CodeFragment(string Id, string Content, string Description);

// Векторное хранилище
public class VectorStore
{
    private readonly OllamaApiClient _ollama;
    private readonly string _embeddingModel;
    private readonly ConcurrentBag<VectorRecord> _vectors = new();

    public VectorStore(OllamaApiClient ollama, string embeddingModel)
    {
        _ollama = ollama;
        _embeddingModel = embeddingModel;
    }

    // Построение хранилища
    public async Task BuildStoreAsync(List<CodeFragment> fragments)
    {
        var tasks = fragments.Select(async fragment =>
        {
            var embeddings = await GetEmbeddingsAsync(fragment.Content);

            foreach (var embedding in embeddings)
            {
                _vectors.Add(new VectorRecord(fragment.Id, fragment.Content, fragment.Description, embedding));
            }
        });

        await Task.WhenAll(tasks);
    }

    // Поиск в хранилище
    public async Task<string> SearchAsync(string query, int topK = 3)
    {
        var queryEmbedding = await GetEmbeddingsAsync(query);
        var results = new List<VectorRecord>();

        foreach (var embedding in queryEmbedding)
        {
            foreach (var vector in _vectors)
            {
                var similarity = CosineSimilarity(embedding, vector.Embedding);
                results.Add(vector with { Similarity = similarity });
            }
        }

        var topResults = results
            .OrderByDescending(r => r.Similarity)
            .Take(topK)
            .ToList();

        return FormatSearchResults(topResults);
    }

    // Получение эмбеддингов
    private async Task<List<float[]>> GetEmbeddingsAsync(string text)
    {
        var response = await _ollama.EmbedAsync(new EmbedRequest
        {
            Model = _embeddingModel,
            Input = new List<string>() { text }
        });
        return response.Embeddings;
    }

    // Расчет косинусного сходства
    private float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length || a.Length == 0)
            return 0;

        float dot = 0.0f, magA = 0.0f, magB = 0.0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB));
    }

    // Форматирование результатов поиска
    private string FormatSearchResults(List<VectorRecord> results)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Релевантный контекст кода:");
        sb.AppendLine("--------------------------------");

        foreach (var (i, result) in results.Select((r, i) => (i, r)))
        {
            sb.AppendLine($"🔍 Совпадение #{i + 1} (точность: {result.Similarity:0.00})");
            sb.AppendLine($"📄 {result.Description}");
            sb.AppendLine(result.Content.Trim());
            sb.AppendLine("--------------------------------");
        }

        return sb.ToString();
    }
}

// Вспомогательные классы
public record VectorRecord(
    string Id,
    string Content,
    string Description,
    float[] Embedding,
    float Similarity = 0
);

// Реализация LINQ-провайдера для исходного кода
public class SourceCodeQueryProvider : IQueryProvider, IDisposable
{
    private readonly string _solutionPath;
    private readonly List<SyntaxTree> _syntaxTrees = new();
    private bool _disposed;

    public SourceCodeQueryProvider(string solutionPath)
    {
        if (string.IsNullOrEmpty(solutionPath))
            throw new ArgumentNullException(nameof(solutionPath));
        if (!Directory.Exists(solutionPath))
            throw new DirectoryNotFoundException($"Путь к решению не найден: {solutionPath}");

        _solutionPath = solutionPath;
        LoadSyntaxTrees();
    }

    private void LoadSyntaxTrees()
    {
        try
        {
            var csFiles = Directory.EnumerateFiles(_solutionPath, "*.cs", SearchOption.AllDirectories);
            foreach (var file in csFiles)
            {
                try
                {
                    var code = File.ReadAllText(file);
                    var syntaxTree = CSharpSyntaxTree.ParseText(code);
                    _syntaxTrees.Add(syntaxTree);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка обработки файла {file}: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка доступа к каталогу {_solutionPath}: {ex.Message}");
            throw;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _syntaxTrees.Clear();
            _disposed = true;
        }
    }

    public IQueryable CreateQuery(Expression expression)
    {
        return new SourceCodeQuery<object>(this, expression);
    }

    public IQueryable<TElement> CreateQuery<TElement>(Expression expression)
    {
        return new SourceCodeQuery<TElement>(this, expression);
    }

    public object Execute(Expression expression)
    {
        return Execute<object>(expression);
    }

    public TResult Execute<TResult>(Expression expression)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(SourceCodeQueryProvider));

        try
        {
            var visitor = new QueryVisitor();
            visitor.Visit(expression);

            var walker = new CustomCSharpSyntaxWalker(visitor.Filters);
            foreach (var tree in _syntaxTrees)
            {
                walker.Visit(tree.GetRoot());
            }

            var results = walker.GetResults();

            // Определение целевого типа
            Type targetType = visitor.Selector?.Method.ReturnType ??
                (typeof(TResult).IsGenericType
                    ? typeof(TResult).GetGenericArguments()[0]
                    : typeof(TResult));

            if (targetType == null)
                throw new InvalidOperationException("Не удалось определить целевой тип");

            var filtered = results
                .Where(n => targetType == typeof(string)
                        || targetType.IsAssignableFrom(n.GetType()))
                .ToList();

            // Применение Select при необходимости
            if (visitor.Selector != null)
            {
                filtered = filtered.Select(node =>
                {
                    try
                    {
                        var result = visitor.Selector.DynamicInvoke(node);
                        return result as CSharpSyntaxNode ??
                            throw new InvalidCastException($"Невозможно преобразовать {result?.GetType()} в CSharpSyntaxNode");
                    }
                    catch (Exception ex)
                    {
                        throw new InvalidOperationException($"Ошибка применения селектора к узлу: {ex.Message}", ex);
                    }
                }).ToList();
            }

            // Применение сортировки
            if (visitor.SortKeySelector != null)
            {
                try
                {
                    var ordered = visitor.IsDescending
                        ? filtered.OrderByDescending(node =>
                            ConvertToComparable(visitor.SortKeySelector.DynamicInvoke(node)))
                        : filtered.OrderBy(node =>
                            ConvertToComparable(visitor.SortKeySelector.DynamicInvoke(node)));

                    filtered = ordered.Cast<CSharpSyntaxNode>().ToList();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Ошибка сортировки: {ex.Message}", ex);
                }
            }

            // Обработка IEnumerable<T>
            if (typeof(TResult).IsGenericType &&
                typeof(TResult).GetGenericTypeDefinition() == typeof(IEnumerable<>))
            {
                var elementType = typeof(TResult).GetGenericArguments()[0];

                // Фильтрация и приведение к целевому типу
                var typedResults = filtered
                    .Where(n => elementType.IsAssignableFrom(n.GetType()))
                    .ToList();

                // Применение Distinct при необходимости
                if (visitor.UseDistinct)
                {
                    typedResults = typedResults
                        .Distinct(new SyntaxNodeEqualityComparer())
                        .ToList();
                }

                // Создание списка нужного типа
                var listType = typeof(List<>).MakeGenericType(elementType);
                var typedList = (IList)Activator.CreateInstance(listType)!;

                foreach (var item in typedResults)
                {
                    if (elementType.IsAssignableFrom(item.GetType()))
                    {
                        typedList.Add(item);
                    }
                }

                return (TResult)typedList;
            }

            // Обработка скалярных результатов
            var scalarResult = filtered.OfType<TResult>().FirstOrDefault();
            if (scalarResult == null && default(TResult) != null)
            {
                throw new InvalidOperationException("Совпадений не найдено");
            }
            return scalarResult!;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Ошибка выполнения запроса: {ex.Message}", ex);
        }
    }

    private static IComparable ConvertToComparable(object? key)
    {
        if (key == null)
            throw new ArgumentNullException(nameof(key), "Ключ не может быть null для сравнения");

        if (key is IComparable comparable)
            return comparable;
        if (key is string str)
            return str;

        var strKey = key.ToString();
        if (strKey == null)
            throw new InvalidOperationException("Невозможно преобразовать ключ в строку");

        return strKey;
    }

    private class SyntaxNodeEqualityComparer : IEqualityComparer<CSharpSyntaxNode>
    {
        public bool Equals(CSharpSyntaxNode? x, CSharpSyntaxNode? y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (x is null || y is null) return false;
            return x.ToFullString() == y.ToFullString();
        }

        public int GetHashCode(CSharpSyntaxNode obj)
        {
            return obj.ToFullString().GetHashCode();
        }
    }
}

public class SourceCodeQuery<T> : IQueryable<T>, IOrderedQueryable<T>
{
    public SourceCodeQuery(SourceCodeQueryProvider provider, Expression? expression = null)
    {
        Provider = provider ?? throw new ArgumentNullException(nameof(provider));
        Expression = expression ?? Expression.Constant(this);
    }

    public Type ElementType => typeof(T);
    public Expression Expression { get; }
    public IQueryProvider Provider { get; }

    public IEnumerator<T> GetEnumerator() =>
        Provider.Execute<IEnumerable<T>>(Expression).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

public class CustomCSharpSyntaxWalker : CSharpSyntaxWalker
{
    private readonly List<Func<CSharpSyntaxNode, bool>> _filters;
    private readonly List<CSharpSyntaxNode> _results = new();

    public CustomCSharpSyntaxWalker(List<Func<CSharpSyntaxNode, bool>> filters)
    {
        _filters = filters;
    }

    public override void VisitClassDeclaration(ClassDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitClassDeclaration(node);
    }

    public override void VisitStructDeclaration(StructDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitStructDeclaration(node);
    }

    public override void VisitInterfaceDeclaration(InterfaceDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitInterfaceDeclaration(node);
    }

    public override void VisitRecordDeclaration(RecordDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitRecordDeclaration(node);
    }

    public override void VisitEnumDeclaration(EnumDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitEnumDeclaration(node);
    }

    public override void VisitDelegateDeclaration(DelegateDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitDelegateDeclaration(node);
    }

    public override void VisitNamespaceDeclaration(NamespaceDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitNamespaceDeclaration(node);
    }

    public override void VisitFileScopedNamespaceDeclaration(FileScopedNamespaceDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitFileScopedNamespaceDeclaration(node);
    }

    public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
    {
        ApplyFilters(node);
        base.VisitMethodDeclaration(node);
    }

    public override void VisitUsingDirective(UsingDirectiveSyntax node)
    {
        ApplyFilters(node);
        base.VisitUsingDirective(node);
    }

    private void ApplyFilters(CSharpSyntaxNode node)
    {
        if (_filters.All(filter => filter(node)))
        {
            _results.Add(node);
        }
    }

    public IEnumerable<CSharpSyntaxNode> GetResults() => _results;
}

public class QueryVisitor : ExpressionVisitor
{
    public List<Func<CSharpSyntaxNode, bool>> Filters { get; } = new();
    public Delegate? Selector { get; private set; }
    public Delegate? SortKeySelector { get; private set; }
    public bool IsDescending { get; private set; }
    public bool UseDistinct { get; private set; }

    protected override Expression VisitMethodCall(MethodCallExpression node)
    {
        if (node.Method.Name == "Distinct")
        {
            UseDistinct = true;
            return base.Visit(node.Arguments[0]);
        }
        else if (node.Method.Name == "Where")
        {
            var lambda = (LambdaExpression)((UnaryExpression)node.Arguments[1]).Operand;
            var filter = CreateFilter(lambda);
            Filters.Add(filter);
        }
        else if (node.Method.Name == "Select")
        {
            var lambda = (LambdaExpression)((UnaryExpression)node.Arguments[1]).Operand;
            Selector = lambda.Compile();
        }
        else if (node.Method.Name == "OrderBy" || node.Method.Name == "OrderByDescending")
        {
            var lambda = (LambdaExpression)((UnaryExpression)node.Arguments[1]).Operand;
            SortKeySelector = lambda.Compile();
            IsDescending = node.Method.Name == "OrderByDescending";
        }

        return base.VisitMethodCall(node);
    }

    private Func<CSharpSyntaxNode, bool> CreateFilter(LambdaExpression lambda)
    {
        var param = lambda.Parameters[0];
        var body = lambda.Body;

        return node =>
        {
            var nodeType = GetNodeType(node);
            if (nodeType != param.Type) return false;

            var condition = Expression.Lambda(body, param)
                .Compile()
                .DynamicInvoke(node);

            return condition is bool b && b;
        };
    }

    private Type GetNodeType(CSharpSyntaxNode node)
    {
        if (node is ClassDeclarationSyntax)
            return typeof(ClassDeclarationSyntax);
        if (node is MethodDeclarationSyntax)
            return typeof(MethodDeclarationSyntax);
        if (node is UsingDirectiveSyntax)
            return typeof(UsingDirectiveSyntax);
        return node.GetType();
    }
}