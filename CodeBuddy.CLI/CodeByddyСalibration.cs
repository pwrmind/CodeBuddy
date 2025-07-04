namespace Сalibration
{
    namespace NestedNamespaceA
    {
        internal class NestedClassA
        {
            public string Echo(string text)
            {
                return text;
            }
        }
    }

    namespace NestedNamespaceB
    {
        public class NestedClassB
        {
            internal class NestedNestedClassB
            {
                NestedNamespaceA.NestedClassA _codeByddy = new NestedNamespaceA.NestedClassA();
                public string Echo(string text)
                {
                    return _codeByddy.Echo(text);
                }
            }

            NestedNestedClassB _codeByddy = new NestedNestedClassB();
            public string Echo(string text)
            {
                return _codeByddy.Echo(text);
            }
        }
    }

    sealed class CodeByddyСalibration
    {
        internal class NestedClassC
        {
            NestedNamespaceB.NestedClassB _codeByddy = new NestedNamespaceB.NestedClassB();
            public string Echo(string text)
            {
                return _codeByddy.Echo(text);
            }
        }
    }
}
