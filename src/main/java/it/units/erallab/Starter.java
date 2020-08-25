/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author eric
 * @created 2020/08/18
 * @project VSREvolution
 */
public class Starter {
  public static void main(String[] args) {
    Pattern p = Pattern.compile("biped-(?<w>\\d++(\\.\\d++)?)x(?<h>[0-9]++)");
    Matcher m = p.matcher("biped-4.11x3");
    System.out.println(m.find());
    System.out.printf("biped-%sx%s%n", m.group("w"), m.group("h"));
  }
}
