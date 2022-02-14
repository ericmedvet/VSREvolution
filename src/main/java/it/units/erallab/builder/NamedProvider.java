package it.units.erallab.builder;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public interface NamedProvider<T> {

  String TOKEN_SEPARATOR = ";";
  String PARAM_VALUE_SEPARATOR = "=";

  T build(String name, Map<String, String> params);

  static <T> NamedProvider<T> empty() {
    return (name, params) -> {
      throw new NoSuchElementException();
    };
  }

  default NamedProvider<T> and(NamedProvider<? extends T> other) {
    NamedProvider<T> thisNamedProvider = this;
    return (name, params) -> {
      try {
        return thisNamedProvider.build(name,params);
      } catch (Throwable ignored) {
      }
      return other.build(name, params);
    };
  }

  default Optional<T> build(String nameAndParams) {
    List<String> pieces = List.of(nameAndParams.split(TOKEN_SEPARATOR));
    String name = pieces.get(0);
    Map<String, String> params = pieces.subList(1, pieces.size()).stream()
        .map(s -> s.split(PARAM_VALUE_SEPARATOR))
        .collect(Collectors.toMap(
            s -> s[0],
            s -> s[1]
        ));
    T t;
    try {
      t = build(name, params);
    } catch (Throwable throwable) {
      return Optional.empty();
    }
    return Optional.of(t);
  }

}
