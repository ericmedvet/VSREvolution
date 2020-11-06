package it.units.erallab.analyzer.expression;

import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Parser {

  //BNF
  //  <e> ::= <n> | <v> | (<e> <o> <e>)

  private final String string;
  private int cursor = 0;

  public Parser(String string) {
    this.string = string.replace(" ", "");
  }

  public enum TokenType {
    CONSTANT("[0-9]+(\\.[0-9]+)?"),
    VARIABLE("[a-z][a-z0-9]*"),
    OPERATOR("[+\\-\\*/\\^]"),
    OPEN_BRACKET("\\("),
    CLOSED_BRACKET("\\)");
    private final String regex;

    TokenType(String regex) {
      this.regex = regex;
    }

    public Token next(String s, int i) {
      Matcher matcher = Pattern.compile(regex).matcher(s);
      if (!matcher.find(i)) {
        return null;
      }
      return new Token(matcher.start(), matcher.end());
    }

    public String getRegex() {
      return regex;
    }
  }

  private static class Token {
    private final int start;
    private final int end;

    public Token(int start, int end) {
      this.start = start;
      this.end = end;
    }
  }

  public Node parse() throws IllegalArgumentException {
    Token token;
    token = TokenType.CONSTANT.next(string, cursor);
    if (token != null && token.start == cursor) {
      cursor = token.end;
      return new Constant(Double.parseDouble(string.substring(token.start, token.end)));
    }
    token = TokenType.VARIABLE.next(string, cursor);
    if (token != null && token.start == cursor) {
      cursor = token.end;
      return new Variable(string.substring(token.start, token.end));
    }
    token = TokenType.OPEN_BRACKET.next(string, cursor);
    if (token != null && token.start == cursor) {
      cursor = token.end;
      Node child1 = parse();
      Token operatorToken = TokenType.OPERATOR.next(string, cursor);
      if (operatorToken != null && operatorToken.start == cursor) {
        cursor = operatorToken.end;
      } else {
        throw new IllegalArgumentException(String.format(
            "Unexpected char at %d instead of operator: '%s'",
            cursor,
            string.charAt(cursor)
        ));
      }
      Node child2 = parse();
      Token closedBracketToken = TokenType.CLOSED_BRACKET.next(string, cursor);
      if (closedBracketToken != null && closedBracketToken.start == cursor) {
        cursor = closedBracketToken.end;
      } else {
        throw new IllegalArgumentException(String.format(
            "Unexpected char at %d instead of closed bracket: '%s'",
            cursor,
            string.charAt(cursor)
        ));
      }
      Operator.Type operatorType = null;
      String operatorString = string.substring(operatorToken.start, operatorToken.end);
      for (Operator.Type type : Operator.Type.values()) {
        if (operatorString.equals(Character.toString(type.getSymbol()))) {
          operatorType = type;
          break;
        }
      }
      if (operatorType == null) {
        throw new IllegalArgumentException(String.format(
            "Unknown operator at %d: '%s'",
            operatorToken.start,
            operatorString
        ));
      }
      return new Operator(operatorType, Arrays.asList(child1, child2));
    }
    throw new IllegalArgumentException(String.format(
        "Unexpected char at %d: '%s'",
        cursor,
        string.charAt(cursor)
    ));
  }

}
